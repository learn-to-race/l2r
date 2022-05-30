import sys
import regex as re
import json
import pandas as pd


# Function to read in data
def read_data(filename):
    return_vals = list()
    with open(filename) as file_handle:
        content = file_handle.readlines()
        for c in content:
            episode = re.findall(r'^\[[A-Z|a-z| |0-9]+\]', c)
            if len(episode) != 0:
                episode = int(
                    episode[0].replace(
                        '[',
                        '').replace(
                        ']',
                        '').replace(
                        'Ep ',
                        ''))
                data_string = '{' + c.split('] {')[1]
                data_string = data_string.replace(
                    "'",
                    "\"").replace(
                    'False',
                    '"False"').replace(
                    'True',
                    '"True"')
                data_dict = json.loads(data_string)
                data_dict['episode'] = episode
                for metric in data_dict['metrics'].keys():
                    data_dict[metric] = data_dict['metrics'][metric]

                del data_dict['metrics']
                return_vals.append(data_dict)
    return return_vals


if __name__ == "__main__":
    filename = sys.argv[1]
    data = read_data(filename)
    df = pd.DataFrame(data)
    output_file = filename.split('.')[0] + ".csv"
    df.to_csv(output_file)
