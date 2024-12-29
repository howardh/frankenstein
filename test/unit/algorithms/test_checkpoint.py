from frankenstein.algorithms.trainer import Checkpoint


"""
Tests to write:
- Frequency: Check that a checkpoint file is created at appropriate intervals
  - Frequency = 1
  - Frequency = 2
  - Frequency = 5
  - Check robustness to skipping steps (e.g. if we save on every 100 steps, then we run `save(98)`, `save(99)`, `save(101)` (skipping step 100), it should save the checkpoint at step 101)
- File name:
  - Default should be "checkpoint.pt"
- Data:
  - ...
- It should load the checkpoint upon initialization
"""


def test_checkpoint_frequency_1(tmp_path):
    # Create a checkpoint object
    checkpoint = Checkpoint(data={}, frequency=(1, 'step'), path=tmp_path)

    # Save the checkpoint
    checkpoint.save(step=0)

    # No file should be created
    assert len(list(tmp_path.iterdir())) == 0

    # Save the Checkpoint
    checkpoint.save(step=1)

    # Check that the file was created
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_frequency_skip(tmp_path):
    # Create a checkpoint object
    checkpoint = Checkpoint(data={}, frequency=(100, 'step'), path=tmp_path)

    for i in [0,10,30,60,90]:
        checkpoint.save(step=i)
    assert len(list(tmp_path.iterdir())) == 0

    # Save the Checkpoint
    checkpoint.save(step=110)

    # Check that the file was created
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_frequency_force_save(tmp_path):
    # Create a checkpoint object
    checkpoint = Checkpoint(data={}, frequency=(100, 'step'), path=tmp_path)

    checkpoint.save(step=0, force=True)

    # Check that the file was created
    assert len(list(tmp_path.iterdir())) == 1
