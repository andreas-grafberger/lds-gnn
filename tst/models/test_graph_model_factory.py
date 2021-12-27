import pytest
from pytest_mock import MockFixture

from src.models.factory import GraphGenerativeModelFactory


@pytest.fixture
def mocked_model_factory(mocker: MockFixture):
    return mocker.MagicMock(GraphGenerativeModelFactory)


def test_raises_exception_on_unknown_method(mocker):
    mock_data = mocker.MagicMock()
    mocked_model_factory = GraphGenerativeModelFactory(mock_data)
    with pytest.raises(NotImplementedError):
        model = mocked_model_factory.create("non_existent_model_type")

# @pytest.mark.parametrize("directed", [(True,), (False,)])
# def test_instantiates_correct_optimizer_for_each_type(mocked_model_factory: GraphGenerativeModelFactory
# ):
#    pass
