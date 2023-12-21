import pytest
from src.BranchyConfig import BranchyConfig


def test_BranchyConfig():
    # test 1
    config = BranchyConfig()
    assert config.self_supervision is False
    assert config.num_branches == -1
    assert config.branch_locations == []

    # test 2
    config = BranchyConfig(
        self_supervision=True, num_branches=2, branch_locations=[1, 2]
    )
    assert config.self_supervision is True
    assert config.num_branches == 2
    assert config.branch_locations == [1, 2]

    # test 3
    with pytest.raises(AssertionError):
        config = BranchyConfig(num_branches=0)

    # test 4
    with pytest.raises(AssertionError):
        config = BranchyConfig(num_branches=-1, branch_locations=[1, 2])

    # test 5
    with pytest.raises(AssertionError):
        config = BranchyConfig(num_branches=2, branch_locations=[1, 2, 3])

    # test 6
    with pytest.raises(AssertionError):
        config = BranchyConfig(num_branches=2, branch_locations=[1])

    # test 7
    with pytest.raises(AssertionError):
        config = BranchyConfig(num_branches=2, branch_locations=1)

    # test 8
    with pytest.raises(AssertionError):
        config = BranchyConfig(
            num_branches=2, branch_locations=[1, 2], self_supervision=1
        )

    # test 9
    with pytest.raises(AssertionError):
        config = BranchyConfig(
            num_branches=2, branch_locations=[1, 2], self_supervision=[]
        )

    # test 10
    with pytest.raises(AssertionError):
        config = BranchyConfig(
            num_branches=2, branch_locations=[1, 2], self_supervision=None
        )

    # test 11
    with pytest.raises(AssertionError):
        config = BranchyConfig(
            num_branches=2, branch_locations=[1, 2], self_supervision="True"
        )

    # test 12
    with pytest.raises(AssertionError):
        config = BranchyConfig(
            num_branches=2, branch_locations=[1, 2], self_supervision=1.0
        )
