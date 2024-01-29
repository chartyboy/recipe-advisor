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
        return {"documents": [["Test"]]}

    @staticmethod
    def query(*args, n_results=0, **kwargs):
        return {"documents": [["Test"] * n_results]}


class StubChromaClient:
    @staticmethod
    def get_or_create_collection(*args, **kwargs):
        return StubChromaCollection()

    def get_collection(*args, **kwargs):
        return StubChromaCollection()


class StubErrorChromaClient:
    @staticmethod
    def get_or_create_collection(*args, **kwargs):
        raise Exception


class StubEmbeddingModel:
    @staticmethod
    def embed_documents(*args, **kwargs):
        return [[1, 2]]


class StubLLM:
    @staticmethod
    def query(*args, **kwargs):
        return "test_reponse"


class MockResponse:
    def __init__(self, json_content=dict(), status_code=200) -> None:
        self.json_content = json_content
        self.status_code = status_code

    def set_json_content(self, json_content):
        self.json_content = json_content

    def json(self):
        return self.json_content
