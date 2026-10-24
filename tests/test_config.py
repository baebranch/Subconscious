import pytest
from subconscious.config import Config, PeerApprovalConfig


def test_config_defaults():
  config = Config()
  assert config.dev is False
  assert "-dev" not in str(config.data_dir)

def test_config_custom():
  config = Config(dev=True)
  assert config.dev is True
  assert "-dev" in str(config.data_dir)

def test_config_peer_defaults():
  config = Config()
  assert config.peer_grace_period_seconds == 30.0
  assert config.peer_approval.query_requires_approval is False
  assert config.peer_approval.mutation_requires_approval is True

def test_config_peer_settings_round_trip(tmp_path):
  config = Config()
  config.data_dir = tmp_path
  config.node_id = "test-node"
  config.secrets = {}
  config.peer_grace_period_seconds = 45.0
  config.peer_approval = PeerApprovalConfig(
    query_requires_approval=True,
    mutation_requires_approval=False,
  )
  config.save()

  loaded = Config()
  loaded.data_dir = tmp_path
  loaded.load()
  assert loaded.peer_grace_period_seconds == 45.0
  assert loaded.peer_approval.query_requires_approval is True
  assert loaded.peer_approval.mutation_requires_approval is False
