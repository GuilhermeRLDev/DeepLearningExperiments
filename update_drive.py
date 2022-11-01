'''
Commit changes to the current repository and pull changes on drive mapping
Making it accesible for google COLAB for execution

@Author: Guilherme Rossetti Lima
'''

import subprocess
import configparser
import os.path
import sys
from os import path
from os import getcwd

PIPE = subprocess.PIPE

def run_command(command, message = None):
    PIPE = subprocess.PIPE
    arr_command = command.split()
    print(arr_command)
    if message is not None:
        arr_command.append(message)

    process = subprocess.Popen(arr_command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    if stderr is not None and stderr != "":
        print(stderr)
    else:
        print(stdout)

def load_configuration(config):
    config_parser = configparser.ConfigParser()
    config_parser.read(config)

    return config_parser["MAIN"]["DRIVE_PATH"]

#The directory where the GoogleDrive is mapped must be updated


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
    os.chdir(drive_location)

    # Pull changes
    print(f"Pulling changes (Syncying might take a while). \r\n")
    run_command(f"git pull")
    run_command(f"git merge")

    # Reflect changes on drive
    print(f"Moving back to project path {current_path}. \r\n")
    os.chdir(current_path)

    #Update completed
    print("Update completed. \r\n")





