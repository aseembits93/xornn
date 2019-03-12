import yaml
with open('config.yaml','r') as f:
    try:
        config = yaml.load(f)
    except yaml.YAMLError as exc:
        print(exc)    
      