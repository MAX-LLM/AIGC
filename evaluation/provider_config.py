"""Independent provider-native settings for SPJ proposal and judge requests."""
from dataclasses import dataclass, field
import math
import os
from urllib.parse import urlsplit

from model_config import load_environment


def _setting(role, suffix, default=None):
    for name in (f'MND_{role.upper()}_{suffix}', f'{role.upper()}_{suffix}'):
        value = os.environ.get(name, '').strip()
        if value:
            return value
    return default


@dataclass(frozen=True)
class NativeModelConfig:
    role: str
    provider: str
    model: str
    effort: str
    max_tokens: int
    api_base: str
    api_key: str = field(repr=False)
    timeout: float = 180.0

    def validate(self, require_model=True):
        if self.role not in {'judge', 'proposal'}:
            raise ValueError('SPJ model role must be judge or proposal.')
        if self.provider not in {'openai', 'anthropic'}:
            raise ValueError('SPJ provider must be openai or anthropic.')
        if require_model and not self.model:
            raise ValueError(f'Set MND_{self.role.upper()}_MODEL or pass --model.')
        if self.effort not in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('Unsupported reasoning effort setting.')
        if self.provider == 'anthropic' and self.effort not in {'low', 'medium', 'high', 'max'}:
            raise ValueError('Anthropic reasoning effort must be low, medium, high or max.')
        if self.max_tokens <= 0 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('SPJ token limit and timeout must be positive.')
        try:
            parsed = urlsplit(self.api_base)
            valid = parsed.scheme in {'http', 'https'} and parsed.netloc and not any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('SPJ API base must be an HTTP(S) URL without credentials, query or fragment.')
        return self

    def public_dict(self):
        self.validate(require_model=False)
        return {key: getattr(self, key) for key in ('role', 'provider', 'model', 'effort', 'max_tokens', 'api_base', 'timeout')}


def native_config(role='judge', *, provider=None, model=None, effort=None, max_tokens=None, api_base=None, api_key=None, timeout=None):
    load_environment()
    if role not in {'judge', 'proposal'}:
        raise ValueError('SPJ model role must be judge or proposal.')
    selected_provider = provider or _setting(role, 'PROVIDER', 'anthropic' if role == 'proposal' else 'openai')
    if selected_provider not in {'openai', 'anthropic'}:
        raise ValueError('SPJ provider must be openai or anthropic.')
    if selected_provider == 'anthropic':
        shared_base = os.environ.get('ANTHROPIC_API_BASE') or os.environ.get('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com'
        shared_key = os.environ.get('ANTHROPIC_API_KEY', '')
    else:
        shared_base = os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
        shared_key = os.environ.get('OPENAI_API_KEY', '')
    try:
        config = NativeModelConfig(
            role=role, provider=selected_provider,
            model=model if model is not None else _setting(role, 'MODEL', ''),
            effort=effort if effort is not None else _setting(role, 'REASONING_EFFORT', _setting(role, 'EFFORT', 'high')),
            max_tokens=int(max_tokens if max_tokens is not None else _setting(role, 'MAX_TOKENS', '8192' if role == 'proposal' else '4096')),
            api_base=api_base if api_base is not None else _setting(role, 'API_BASE', shared_base),
            api_key=api_key if api_key is not None else _setting(role, 'API_KEY', shared_key),
            timeout=float(timeout if timeout is not None else _setting(role, 'TIMEOUT', '180')),
        )
    except (TypeError, ValueError):
        raise ValueError(f'Invalid SPJ {role} numeric setting.') from None
    return config.validate(require_model=False)


def model_role(manifest=None, override=None):
    role = override or (manifest or {}).get('model_role') or ('proposal' if (manifest or {}).get('generator') else 'judge')
    if role not in {'judge', 'proposal'}:
        raise ValueError('SPJ batch model role must be judge or proposal.')
    return role


def make_client(provider, role, *, api_base=None, api_key=None, timeout=None, workspace_id=None):
    config = native_config(role, provider=provider, api_base=api_base, api_key=api_key, timeout=timeout)
    if not config.api_key:
        raise ValueError(f'Set MND_{role.upper()}_API_KEY or the provider API key before making requests.')
    kwargs = {'api_key': config.api_key, 'timeout': config.timeout, 'max_retries': 0}
    if provider == 'openai':
        from openai import OpenAI
        return OpenAI(base_url=config.api_base.rstrip('/'), **kwargs)
    from anthropic import Anthropic
    selected_workspace = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')
    if selected_workspace:
        kwargs['default_headers'] = {'anthropic-workspace-id': selected_workspace}
    return Anthropic(base_url=config.api_base.rstrip('/').removesuffix('/v1'), **kwargs)
