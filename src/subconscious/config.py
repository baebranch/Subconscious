import os
import sys
import uuid
import yaml
import json
import logging
import pathlib
import keyring
from typing import Optional
from cryptography.fernet import Fernet
from dataclasses import dataclass, field


# Logging setup
logger = logging.getLogger("subconscious")

# Secrets Setup
SUBCONSCIOUS_KEY = keyring.get_password("subconscious", "encryption_key")
if not SUBCONSCIOUS_KEY:
  SUBCONSCIOUS_KEY = Fernet.generate_key().decode()
  keyring.set_password("subconscious", "encryption_key", SUBCONSCIOUS_KEY)
CIPHER = Fernet(SUBCONSCIOUS_KEY.encode())
LOGO = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣶⣶⣦⣠⣤⣤⣤⣄⣀⣠⣤⣤⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⠏⠉⠉⠉⠙⢻⣿⣿⣿⡷⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⠋⠙⠛⠻⣷⣄⣀⠀⣠⣾⠟⠛⠛⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣰⣾⣿⣷⣄⣀⣀⣀⣹⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⠟⠛⠛⠛⠻⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣤⣀⠀⠀⠀⢀⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⠿⣶⣶⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⢿⣶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣰⡿⠉⠀⠀⠈⠙⢿⣦⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣷⡀⠀⠀⠀⠀⠈⣿⡆⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⠿⠿⣶⣦⣄⣀⣤⣿⣇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣀⣀⣠⣾⠟⠁⢸⣿⠀⠀⠀⠉⠙⣿⣿⣿⣿⠆⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⠁⠀⢀⣼⣿⡀⠀⠀⢀⣠⣿⠿⠟⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⠿⣿⡿⠿⠿⢷⣿⣿⣿⣿⣶⠿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

def get_default_data_dir() -> pathlib.Path:
  """ Returns the default data directory based on the OS. """
  if sys.platform == "win32":
    return pathlib.Path(os.environ.get("APPDATA", "~")).expanduser() / "Subconscious"
  elif sys.platform == "darwin":
    return pathlib.Path.home() / "Library" / "Application Support" / "Subconscious"
  else:
    # Linux/Unix default to XDG_CONFIG_HOME or ~/.config
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
      return pathlib.Path(xdg_config) / "subconscious"
    return pathlib.Path.home() / ".config" / "subconscious"


@dataclass
class PeerApprovalConfig:
  """ This Engine's own gate for tool calls it is asked to execute on behalf
      of a Peer_Engine — configured locally, never influenced by the
      requesting Master_Engine. """

  query_requires_approval: bool = False
  mutation_requires_approval: bool = True


@dataclass
class Config:
  api: bool = True
  dev: bool = False
  tui: bool = False
  gui: bool = False
  node_id: Optional[str] = None
  secrets: Optional[dict] = None
  data_dir: pathlib.Path = field(default_factory=get_default_data_dir)
  peer_grace_period_seconds: float = 30.0
  peer_approval: PeerApprovalConfig = field(default_factory=PeerApprovalConfig)

  def __post_init__(self):
    if self.dev:
      self.data_dir = get_default_data_dir()
      self.data_dir = self.data_dir.with_name(f"{self.data_dir.name}-dev")

  def load(self):
    """ Loads config from the YAML file. """
    path = self.data_dir / "config.yaml"
    if path.exists():
      with open(path, 'r') as f:
        data = yaml.safe_load(f)
        if data:
          self.node_id = data.get('node_id', self.node_id)
          self.secrets = data.get('secrets', self.secrets)
          self.peer_grace_period_seconds = data.get('peer_grace_period_seconds', self.peer_grace_period_seconds)
          peer_approval = data.get('peer_approval')
          if peer_approval:
            self.peer_approval = PeerApprovalConfig(
              query_requires_approval=peer_approval.get('query_requires_approval', self.peer_approval.query_requires_approval),
              mutation_requires_approval=peer_approval.get('mutation_requires_approval', self.peer_approval.mutation_requires_approval),
            )
    else:
      # Create a config file
      logger.info(f"No config file found at {path}. Creating one...")
      self.node_id = str(uuid.uuid4())
      self.secrets = {}
      self.save()
      logger.info(f"Config file created at {path}.")

  def save(self):
    """ Saves config to the YAML file. """
    path = self.data_dir / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
      yaml.safe_dump({
        'node_id': str(self.node_id),
        'secrets': self.secrets,
        'peer_grace_period_seconds': self.peer_grace_period_seconds,
        'peer_approval': {
          'query_requires_approval': self.peer_approval.query_requires_approval,
          'mutation_requires_approval': self.peer_approval.mutation_requires_approval,
        },
      }, f)

  @property
  def db_path(self) -> str:
    """Returns the path to the SQLite database."""
    return f"sqlite+aiosqlite:///{self.data_dir / 'subconscious.db'}"
  
  async def write_keyring(self) -> None:
    """ Update the keyring with the new secrets """
    encrypted = CIPHER.encrypt(json.dumps(self.secrets).encode())
    with open(f'{self.data_dir}/data.enc', 'wb') as f:
      f.write(encrypted)

  def read_keyring(self) -> None:
    """ Read the keyring for the secrets """
    # Check if the keyring file exists
    if os.path.exists(f'{self.data_dir}/data.enc'):
      with open(f'{self.data_dir}/data.enc', 'rb') as f:
        encrypted = f.read()
        decrypted = CIPHER.decrypt(encrypted).decode()
    # Else set to none
    else:
      decrypted = None

    # Return the decrypted settings or default values
    self.secrets = json.loads(decrypted) if decrypted else {
      "models": {},
      "tools": {},
      "mcp": {},
      "_thread_tools": {}
    }


def log_config(config: Config):
  """ Logs the current configuration settings. """
  print("-" * 40)
  if config.tui:
    print("Mode: Engine + CLI")
  if config.gui:
    print("Mode: Engine + GUI")
  else:
    print("Mode: Engine")
  print(f"Live: {not config.dev}")
  print(f"Node ID: {config.node_id}")
  print(f"Data Directory: {config.data_dir}")
  print("-" * 40)
