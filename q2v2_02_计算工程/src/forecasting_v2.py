"""PDF §3.2–3.3：因果预测、配对整日残差、有放回抽样。

P2 使用日周期傅里叶回归 + ARIMA(1,0,0) 残差；P3 使用随机森林。
所有拟合函数只接收已完成日期，禁止把目标日实际值作为特征。
"""
from datetime import timedelta
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from forecasting import point_forecast

MODELS = ('P1_weighted28', 'P2_fourier_arima', 'P3_random_forest')


def fourier(t, harmonics=8):
    t = np.asarray(t)
    angle = 2 * np.pi * t[:, None] * np.arange(1, harmonics + 1) / 144
    return np.c_[np.ones(len(t)), np.sin(angle), np.cos(angle)]


def arima_forecast(history):
    """先估计确定日周期，再对残差拟合低阶平稳 ARIMA；不做全样本调参。"""
    y = np.asarray(history[-28:], dtype=float).ravel()
    design = fourier(np.arange(len(y)))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coef
    if np.std(residual) < 1e-8:
        phi, correction, diagnostics = 0., np.zeros(144), []
    else:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fit = ARIMA(residual, order=(1, 0, 0), trend='n').fit(method='burg')
        phi = float(fit.arparams[0])
        correction = np.asarray(fit.forecast(144))
        diagnostics = [str(w.message) for w in captured]
    prediction = fourier(np.arange(len(y), len(y) + 144)) @ coef + correction
    return np.maximum(prediction, 0), {'ar1': phi, 'harmonics': 8,
        'fit': 'Fourier OLS then ARIMA(1,0,0), Burg estimation', 'warnings': diagnostics}


def tree_features(history, day, target_date):
    """day 可等于 len(history)；只索引 day-1 及更早的整日曲线。"""
    if day < 7 or day > len(history):
        raise ValueError('Tree features require seven completed days')
    previous = history[day-1]
    period = np.arange(144)
    clock = fourier(period, 3)[:, 1:]
    calendar = np.tile([np.sin(2*np.pi*target_date.weekday()/7),
                        np.cos(2*np.pi*target_date.weekday()/7),
                        np.sin(2*np.pi*(target_date.month-1)/12),
                        np.cos(2*np.pi*(target_date.month-1)/12)], (144, 1))
    return np.c_[clock, calendar, previous, history[day-7], history[day-7:day].mean(axis=0),
                 np.roll(previous, 1), np.roll(previous, -1)]


def tree_forecast(history, history_dates, target_date, seed):
    end = len(history)
    begin = max(7, end-28)
    x = np.vstack([tree_features(history, j, history_dates[j]) for j in range(begin, end)])
    y = history[begin:end].ravel()
    reg = RandomForestRegressor(n_estimators=64, max_depth=10, min_samples_leaf=8,
                                max_features=1., random_state=seed, n_jobs=1)
    reg.fit(x, y)
    return np.maximum(reg.predict(tree_features(history, end, target_date)), 0)


def predict(history_load, history_pv, history_dates, target_date, model, prior_load, prior_pv, seed):
    n = len(history_dates)
    if model not in MODELS:
        raise ValueError('Unknown candidate')
    if n:
        if history_dates[-1] != target_date-timedelta(days=1) or any(d >= target_date for d in history_dates):
            raise ValueError('Future observations or missing latest completed day')
        if any(b-a != timedelta(days=1) for a, b in zip(history_dates, history_dates[1:])):
            raise ValueError('History must be consecutive')
    info = {'model': model, 'origin': target_date.isoformat(),
            'training_end': history_dates[-1].isoformat() if n else None,
            'training_start': history_dates[max(0,n-28)].isoformat() if n else None,
            'history_days': min(n, 28)}
    if n == 0:
        info['implementation'] = 'attachment1_prior_PDF_3.9.4'
        return np.asarray(prior_load).copy(), np.asarray(prior_pv).copy(), info
    if model == MODELS[0] or n < (7 if model == MODELS[1] else 8):
        l, g, meta = point_forecast(history_load, history_pv, history_dates, target_date, 'weighted28')
        info['implementation'] = 'weighted28' if model == MODELS[0] else 'short_history_weighted28_fallback'
        info['weights'] = meta['weights']
        return l, g, info
    if model == MODELS[1]:
        l, il = arima_forecast(history_load)
        g, ig = arima_forecast(history_pv)
        info.update(implementation='Fourier_plus_ARIMA_errors', load_fit=il, pv_fit=ig)
    else:
        l = tree_forecast(history_load, history_dates, target_date, seed)
        g = tree_forecast(history_pv, history_dates, target_date, seed)
        # 最早训练行还引用其前7天，因此记录的原始观测边界再向前7天。
        info['training_start'] = history_dates[max(0, n-35)].isoformat()
        info.update(implementation='separate_random_forests', n_estimators=64,
                    max_depth=10, min_samples_leaf=8, training_target_days=min(n-7,28))
    if not np.isfinite(l).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite forecast')
    return l, g, info


def build_cache(dataset, model, seed=20260912, stop=365, progress=None):
    l, g = np.full_like(dataset.load_kw, np.nan), np.full_like(dataset.pv_kw, np.nan)
    metadata = {}
    for day in range(stop):
        l[day], g[day], metadata[day] = predict(dataset.load_kw[:day], dataset.pv_kw[:day],
            dataset.dates[:day], dataset.dates[day], model, dataset.baseline_load_kw,
            dataset.baseline_pv_kw, seed)
        if progress and (day == 0 or (day+1) % 60 == 0 or day+1 == stop):
            progress(f'{model}: forecasts {day+1}/{stop}')
    return {'model': model, 'load_kw': l, 'pv_kw': g, 'metadata': metadata}


def scenarios(dataset, cache, day, *, seed=20260912, draws=256, residual_days=28,
              residual_scale=1., mode='bootstrap'):
    """同一抽样日索引同时作用于144点负荷和144点光伏。

重复抽到同一天时合并变量块，概率为抽中次数/draws，与未压缩模型严格等价。
仅按式(25)截断负值，不叠加历史支撑范围遮罩。
"""
    if mode not in ('bootstrap', 'enumerate', 'point'):
        raise ValueError('Unknown scenario rule')
    if day < 0 or day >= len(dataset.dates) or draws < 1 or residual_days < 1 or residual_scale < 0:
        raise ValueError('Invalid scenario configuration')
    pl, pg = cache['load_kw'][day], cache['pv_kw'][day]
    audit = {**cache['metadata'][day], 'scenario_method': mode, 'seed': seed,
             'draw_count': draws if mode == 'bootstrap' and day else 0,
             'residual_scale': residual_scale}
    if day == 0 or mode == 'point':
        ids, counts = np.array([], dtype=int), np.array([1])
        raw_l, raw_g = pl[None, :], pg[None, :]
        prob = np.array([1.])
    else:
        pool = np.arange(max(0, day-residual_days), day)
        if mode == 'bootstrap':
            rng = np.random.default_rng(np.random.SeedSequence([seed, day]))
            ids, counts = np.unique(rng.choice(pool, size=draws, replace=True), return_counts=True)
        else:
            ids, counts = pool, np.ones(len(pool), dtype=int)
        raw_l = pl + residual_scale*(dataset.load_kw[ids]-cache['load_kw'][ids])
        raw_g = pg + residual_scale*(dataset.pv_kw[ids]-cache['pv_kw'][ids])
        prob = counts/counts.sum()
    audit.update(residual_dates=[dataset.dates[j].isoformat() for j in ids],
                 residual_counts=counts.tolist(), scenario_count=len(prob),
                 negative_load_clipped=int((raw_l < 0).sum()), negative_pv_clipped=int((raw_g < 0).sum()))
    l, g = np.maximum(raw_l, 0)/6, np.maximum(raw_g, 0)/6
    if not np.isfinite(l).all() or not np.isfinite(g).all():
        raise ValueError('Forecast cache is incomplete')
    return l, g, prob, audit
