from action_provider.action_provider_dds import DDSActionProvider


from action_provider.action_provider_replay import FileActionProviderReplay

from action_provider.action_provider_wh_dds import DDSRLActionProvider
from pathlib import Path

from action_provider.action_provider_gr00t import GR00TActionProvider


def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        return DDSActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "dds_wholebody":
        return DDSRLActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env,args_cli=args)
    elif args.action_source == "gr00t": # 添加GR00T action provider
        try:
            GR00T_AVAILABLE = True
            print("🔍 Checking for GR00T action provider availability...")
        except ImportError as e:
            print(f"⚠️ GR00T action provider not available: {e}")
            GR00T_AVAILABLE = False

        if not GR00T_AVAILABLE:
            print("❌ GR00T action provider is not available")
            return None
        print("🚀 Initializing GR00T action provider...")
        return GR00TActionProvider(env, args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None