def get_engine_status(predicted_rul: float) -> str:

    if predicted_rul > 50:
        return "Healthy"

    elif predicted_rul > 20:
        return "Degrading"

    return "Critical"


def get_sensor_trends(recent_sensor_data):

    if not recent_sensor_data:
        return "No sensor trend data available."

    observations = []

    for sensor, values in recent_sensor_data.items():

        if len(values) < 2:
            continue

        first = float(values[0])
        last = float(values[-1])

        if first == 0:
            observations.append(
                f"{sensor}: trend could not be calculated."
            )
            continue

        change = ((last - first) / abs(first)) * 100

        direction = "increased" if change > 0 else "decreased"

        observations.append(
            f"{sensor} {direction} by {abs(change):.2f}% "
            f"over the recent analysis window."
        )

    if not observations:
        return "No usable sensor trend information."

    return "\n".join(observations[:10])


def get_feature_importance(feature_importance):

    if not feature_importance:
        return "No SHAP feature importance available."

    lines = []

    for item in feature_importance[:10]:

        feature = item.get(
            "feature",
            "unknown"
        )

        importance = float(
            item.get(
                "mean_abs_shap",
                0
            )
        )

        lines.append(
            f"{feature}: {importance:.4f}"
        )

    return "\n".join(lines)