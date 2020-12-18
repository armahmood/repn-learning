import sys
import json
import itertools
import os

def read_config():
    with open("master_config.json", 'r') as json_file: 
      data = json.load(json_file)
    return data

def main():
    config = read_config()
    try:
      assert os.path.exists("cfg_temp/")
    except:
      os.makedirs('cfg_temp/')
    i=0
    for param_vals in itertools.product(*config.values()):
        cfg = dict(zip(config.keys(), param_vals))
        config_struct ={
            "examples" : cfg["examples"],
            "runs" : 1,
            "input_size" : cfg["input_size"],
            "features" : [cfg["features"]],
            "learner_seeds": [cfg["learner_seeds"]],
            "target_seed" : cfg["target_seed"],
            "activation_target" : cfg["activation_target"],
            "activation_learning" : cfg["activation_learning"],
            "tester" : cfg["tester"],
            "tester_lr" : cfg["tester_lr"],
            "replacement_rate": cfg["replacement_rate"],
            "step": cfg["step"]
        }
        jsonstr = json.dumps(config_struct)
        w_name = 'cfg_temp/'+ str(i) + '.json'
        i = i+1
        with open(w_name, "w") as outfile: 
            outfile.write(jsonstr) 


if __name__ == '__main__':
    main()
