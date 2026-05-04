from .basic_metrics import basic_metricor, generate_curve

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}
    grader = basic_metricor()

    # Threshold Independent
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)

    # Threshold Dependent (pred=None이면 내부에서 threshold sweep으로 best 값 사용)
    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)

    # Affiliation은 affiliation 모듈이 없으면 터질 수 있어서 안전하게 처리
    try:
        Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    except Exception:
        Affiliation_F = float("nan")

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    return metrics
