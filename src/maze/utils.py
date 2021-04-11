def stringify_dict(d: dict) -> str:
    return ", ".join(":".join((str(k), str(v))) for k, v in d.items())


def preprocess_hyperparameters_filename(d: dict) -> str:
    return "__".join("_".join((str(k), str(v))) for k, v in d.items())
