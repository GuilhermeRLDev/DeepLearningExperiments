'''
@Author: Guilherme Rossetti Lima
'''

import subprocess
import configparser
import os.path
import sys
from os import path
from os import getcwd

PIPE = subprocess.PIPE

def load_configuration(config):
    config_parser = configparser.ConfigParser()
    config_parser.read(config)
    print(config_parser)

    return config_parser["MAIN"]["DRIVE_PATH"]

#The directory where the GoogleDrive is mapped must be updated
def run_command(command, message = None):
    arr_command = command.split()

    if message is not None:
        arr_command.append(message)

    process = subprocess.Popen(arr_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if stderr is not None and stderr is not "":
        print(stderr)
    else:
        print(stdout)

if __name__ == "__main__":

    branch = None

    #update commit message
    if len(sys.argv) > 1:
        message = sys.argv[1]

    #update branch
    if len(sys.argv) > 2:
        branch = sys.argv[2]

    current_path = getcwd()
    print(f"{current_path}/credentials.ini")
    drive_location = load_configuration(f"credentials.ini")

    #Update current branch
    print("Pulling current branch. \r\n")
    run_command(f"git pull")

    #Commit changes
    print("Commiting changes. \r\n")
    run_command(f"git add .")
    print(f'git commit -m {message}')
    run_command(f"git commit -m", message)
    print("Pushing changes. \r\n")
    run_command(f"git push")

    # Move to drive
    print(f"Moving to {drive_location}. \r\n")
    run_command(f"cd {drive_location}")

    # Pull changes
    print(f"Pulling changes (Syncying might take a while). \r\n")
    run_command(f"git pull")

    # Reflect changes on drive
    print(f"Moving back to project path {current_path}. \r\n")
    run_command(f"cd {current_path}")

    print("Update completed. \r\n")




