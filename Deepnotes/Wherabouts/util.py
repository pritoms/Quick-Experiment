def build_data_loader(datasets):
    for input, label in datasets:
        yield {
            "input": input,
            "label": label
        }
