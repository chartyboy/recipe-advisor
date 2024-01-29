from dependency_injector import containers, providers


class TestContainer(containers.DeclarativeContainer):
    test_provider = providers.Object(123)


class OverrideContainer(containers.DeclarativeContainer):
    test_provider = providers.Object(1)
