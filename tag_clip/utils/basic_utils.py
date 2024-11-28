import argparse
import toml

def read_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_path:
        return args
    
    config_path = args.config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
        
    config_dict = {}
    for section_name, section_dict in config.items():
        if not isinstance(section_dict, dict):
            config_dict[section_name] = section_dict
            continue

        for key, value in section_dict.items():
            config_dict[key] = value

    config_args = argparse.Namespace(**config_dict)
    args = parser.parse_args(namespace=config_args)
    return args

def read_config_ipynb(parser, config_path):
    args = parser.parse_args([])

    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
        
    config_dict = {}
    for section_name, section_dict in config.items():
        if not isinstance(section_dict, dict):
            config_dict[section_name] = section_dict
            continue

        for key, value in section_dict.items():
            config_dict[key] = value

    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args
