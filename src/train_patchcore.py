from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    datamodule = Folder(
        name="bottle",
        root="data/bottle",
        normal_dir="train/good",
        abnormal_dir="test",
        normal_test_dir="test/good",
        val_split_mode="from_test",
        val_split_ratio=0.2,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
    engine.export(model=model, export_type="torch")

    test_results = engine.test(model=model, datamodule=datamodule)
    print(test_results)


if __name__ == "__main__":
    main()