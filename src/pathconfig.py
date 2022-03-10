import typing as t
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PathConfig:
    base_path: t.Optional[Path] = Path(__file__).absolute().parent.parent
    configs: t.Optional[Path] = None
    models: t.Optional[Path] = None
    appdata: t.Optional[Path] = None
    tokenizers: t.Optional[Path] = None

    def __post_init__(self):
        self.configs = self.base_path.joinpath("configs")
        self.models = self.base_path.joinpath("models")
        self.appdata = self.base_path.joinpath("appdata")
        if self.models:
            self.tokenizer = self.models.joinpath("tokenizer")

    def to_dict(self):
        paths = asdict(self)
        paths = {k: str(v) for k, v in paths.items()}
        return paths


if __name__ == "__main__":
    path_repo = PathConfig()
    print(path_repo.to_dict())
