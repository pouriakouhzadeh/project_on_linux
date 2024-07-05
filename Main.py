import time
import sys
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
from Desision_class import DesisionClass
from unzip import  UNZIP
import glob
import shutil
import concurrent.futures
from print_pourias_expert import BlinkingTextPrinter

CSV_DIRECTORY = "/home/pouria/project/csv_files_for_main_program/currency_data.zip"
extract_directory = "/home/pouria/project/csv_files_for_main_program/"
directory_path = '/home/pouria/project/csv_files_for_main_program/'
directory_path_models = '/home/pouria/project/trained_models/'
directory_path_answer = '/home/pouria/project/answers/'
extension = '.csv'
extension_answer = '.txt'
extension_acn = '.acn'
extension_model = '_for_train.pkl'

def copy_files(source_folder, destination_folder):
    n = 0
    for file_path in glob.glob(os.path.join(source_folder, '*.csv')):
        file_name = os.path.basename(file_path)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy2(file_path, destination_file)
        n += 1
    print(f"\r{n} CSV files copied successfully")

def df_to_json(df):
    # Setting double precision to 15 to maintain floating point accuracy
    return df.to_json(orient='split', double_precision=15)

def json_to_df(json_data):
    return pd.read_json(json_data, orient='split')

def time_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    time_left = next_hour - now
    return time_left.total_seconds()


def delete_acns():
    files = os.listdir(directory_path_answer)
    if len(files) == 11 :
         time.sleep(1)
         for count in range (11) :
            temp = files[count]
            # print(temp[-12:])
            if temp[-3:] == "acn" :
                os.remove(directory_path_answer+files[count])
                print(f"\r Deleting ACN files : {count}", end="", flush=True)

def delete_TXT_files():
    files = os.listdir(directory_path_answer)
    print(f"\rNumber of file remaining if ANSWER directory is : {len(files)}", end="", flush=True)
    files1 = os.listdir(directory_path_answer)
    try :
        if len(files) == 22  :
            time.sleep(1)
            while len(files1) >= 1 :
                os.remove(directory_path_answer+files[len(files1)-1])
                # time.sleep(2)
                files1 = os.listdir(directory_path_answer)
                print(f"\r Deleting ACN and TXT files {len(files1)}", end="", flush=True)
    except:
        print(f"Can not delete files")

def process_item(item, directory_path, directory_path_answer):
    Answer = [0,0]
    position = "NAN"
    try:
        with open(f"/home/pouria/project/trained_models/{item}_parameters.pkl", 'rb') as f:
            ind = pickle.load(f)
        if ind[0] != 0 and ind[1] != 0 and ind[2] != 0:
            print(f"Start to predicting {item[:-2]} :")
            print(f"reading {item}")
            data = pd.read_csv(directory_path+item+'.csv')
            df1 = df_to_json(data)
            df2 = json_to_df(df1)
            print("Runing Desision class for forcasting ...")
            position, Answer = DesisionClass().Desision(df2, item[:-2], ind[:-1], ind[-1])
            print(f"Position = {position}")
            print(f"Answer = {Answer}")
            print("remove csv file")
            print("-----------------------------------")
            os.remove(directory_path+item+'.csv')
            with open(directory_path_answer + item + '.txt', 'w') as file:
                file.write(position+"\n")
                file.write(str(Answer[0])+"\n")
                file.write(str(Answer[1]))
        else:
            print(f"\rThere is no saved model for {item} so delete csv file", end="", flush=True)
            os.remove(directory_path+item+'.csv')
            with open(directory_path_answer + item + '.txt', 'w') as file:
                file.write(position+"\n")
                file.write(str(Answer[0])+"\n")
                file.write(str(Answer[1]))   
    except Exception as e:
        print(f"Error in Desition {item}: {e}")

def switch_case(value):
    if value == 1:
        print(f"\rExpert is working {int(time_until_next_hour()/60)} min to next hour | ", end="", flush=True)
    elif value == 2:
        print(f"\rExpert is working {int(time_until_next_hour()/60)} min to next hour / ", end="", flush=True)
    elif value == 3:
        print(f"\rExpert is working {int(time_until_next_hour()/60)} min to next hour --", end="", flush=True)
    elif value == 4:
        print(f"\rExpert is working {int(time_until_next_hour()/60)} min to next hour \\ ", end="", flush=True)
    else:
        print("Invalid case", end="", flush=True)

unzipper = UNZIP()
print_count = 1
blinking_printer = BlinkingTextPrinter("POURIA'S EXPERT")
blinking_printer.print_text(duration=1, interval=0.5)
print(" ")
count = 0
while True:

    unzipper.unzip_file(CSV_DIRECTORY, extract_directory)
    files = os.listdir(directory_path)
    # print(f" Number of CSV files wating for making answer is :  {len(files)}")
    New_currency_list = []
    if len(files) == 11 :
            
        files_with_creation_time = []
        for filename in os.listdir(directory_path):
                if filename.endswith(extension):
                    file_path = os.path.join(directory_path, filename)
                    creation_time = os.path.getctime(file_path)
                    files_with_creation_time.append((file_path, creation_time))
        files_with_creation_time.sort(key=lambda x: x[1])

        New_currency_list = []
        for file_path, creation_time in files_with_creation_time:
            Answer = [0,0]
            position = "NAN"    
            try :
                with open(f"/home/pouria/project/trained_models/{file_path[-12:-4]}_parameters.pkl", 'rb') as f:
                    ind = pickle.load(f)
                if ind[0] != 0 and ind[1] != 0 and ind[2] != 0 :
                    New_currency_list.append(file_path[-12:-4])            
                if ind[0] == 0 and ind[1] == 0 and ind[2] == 0 :
                    raise Exception("All ind == 0")
            except :
                print(f"\rError in reading some files related to : {file_path[-12:-6]} so delete csv file", end="", flush=True)
                with open(directory_path_answer + file_path[-12:-4] + '.txt', 'w') as file:
                    file.write(position+"\n")
                    file.write(str(Answer[0])+"\n")
                    file.write(str(Answer[1]))
                os.remove(file_path)


        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, item, directory_path, directory_path_answer) for item in New_currency_list]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        
             
    time.sleep(1)
    delete_acns()

    base_counter = 100

    while base_counter > 0 :
        files = os.listdir(directory_path_answer)
        if len(files) == 11 :   
            print("\rNew files for calculating just recived progress broken to coninue proccecing", end="", flush=True)
            delete_acns()
            print(f"\r                                                                                    ", end="", flush=True)
            break
        files = os.listdir(directory_path_answer)            
        if len(files) == 22 :   
            print("\rNew files for calculating just recived progress broken to coninue proccecing", end="", flush=True)
            delete_TXT_files()
            delete_acns()
            print(f"\r                                                                                    ", end="", flush=True)
            break     
        
        files = os.listdir(directory_path)
        if len(files) == 11 :   
            break       
        time.sleep(1)
        count += 1
        base_counter -= 1
        files = os.listdir(directory_path_answer)
        # print(f"Number of file remaining if ANSWER directory is : {len(files)}")
        if len(files) == 0 :
            base_counter = 0

        if len(files) > 0 and len(files) != 11 and len(files) != 22 and count > 60:
            print("\rThere is problem about answer and acn files", end="", flush=True)
            delete_TXT_files()
            delete_acns()                
            count = 0 
            # remaining_time = 0

    switch_case(print_count)
    print_count += 1
    if print_count > 4 :
        print_count = 1

    if len(New_currency_list) > 0 :
        blinking_printer = BlinkingTextPrinter("POURIA'S EXPERT")
        blinking_printer.print_text(duration=1, interval=0.5) 
        print(" ")
# print("Next hour has started!")
