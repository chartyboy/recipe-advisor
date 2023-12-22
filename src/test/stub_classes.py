class StubDocument:
    def __init__(self) -> None:
        self.page_content = "TEST"


class StubJSONLoader:
    @staticmethod
    def load(*args, **kwargs):
        return [StubDocument()]


class StubChromaCollection:
    @staticmethod
    def add(*args, **kwargs):
        pass

    @staticmethod
    def get(*args, **kwargs):
        pass


class StubChromaClient:
    @staticmethod
    def get_or_create_collection(*args, **kwargs):
        return StubChromaCollection()

    def get_collection(*args, **kwargs):
        return StubChromaCollection
