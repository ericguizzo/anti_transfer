
from __future__ import print_function
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import xval_instance as xval

#EXPERIMENT PARAMETERS:
gpu_ID = 0
overwrite_results = False  #if true overwrite existing experiment instances
debug_mode = True  #if false, if an error occurs in one instance, it is skipped without stopping the routine
short_description = 'baseline with no antitransfer, but with weight initialization, iemocap speaker-wise split pretrained on iemocap'
dataset = 'iemocap_actorsplit_spectrum_fast'
task_type = 'classification'
generator = True
num_experiment = 13  #id of the experiment
num_folds = 1  #number of k-folds for cross-validation
#experiment_folder = '../../../copy/prova_API'  #where to save results
experiment_folder = '../results_prova'  #where to save results

global_parameters = ['output_classes=4',
                     'batch_size=30',
                     'architecture="vgg16"',
                     'save_model_metric="loss"',
                     'reshaping_type="cnn"'
                     ]

#DEFINE HERE EVERY INSTANCE OF THE EXPERIMENT
#every instance must be a key in the experiment dict
#every key must be a list of strings
#every parameter overwrites the correspective default parameter
#mandatory parameters:
#-task_type: classification, or regression
#-reshaping type: conv, lstm, none
#-architecture: one of the models defined in the models_API script
#-comment_1 and comment_2: write here any info you want to show in the results spreadsheet. Example: L2 increased to 0.1
experiment = {}


#WRITE HERE EXPERIMENTS

experiment[1] = ['comment_1="no antitransfer"', 'comment_2="weight init iemocap"',
                 'anti_transfer=False', 'at_pretraining="iemocap"',
                 ]

experiment[2] = ['comment_1="no antitransfer"', 'comment_2="weight init iemocap"',
                 'anti_transfer=False', 'at_pretraining="iemocap"',
                 ]

experiment[3] = ['comment_1="no antitransfer"', 'comment_2="no anti, pretraining iemocap"',
                 'anti_transfer=False', 'at_pretraining="iemocap"',
                 ]







#DON'T TOUCH WHAT IS WRITTEN BELOW THIS LINE
#-------------------------------------------------------------------------------#
#outer arguments
try:
    begin = eval(sys.argv[1])
    end = int(sys.argv[2])
    gpu_ID = int(sys.argv[3])

except IndexError:
    #if not specified run all experiments
    keys = list(experiment.keys())
    begin = keys[0]
    end = keys[-1]

#add vand update global parameters
try:
    added_global_parameters = sys.argv[4]
    added_global_parameters = added_global_parameters.split('%')
    added_global_parameters = list(filter(lambda x: x != '', added_global_parameters))
    if len(added_global_parameters) > 0:
        for added in added_global_parameters:  #iterate global parameters
            flag = True
            added_name = added.split('=')[0]
            for i in range(len(global_parameters)):
                global_name = global_parameters[i].split('=')[0]
                if global_name == added_name:
                    global_parameters[i] = added
                    flag = False

            if flag == True:
                global_parameters.append(added)
except:
    pass


#update xval instance parameters
try:
    xval_global_parameters = sys.argv[5]
    xval_global_parameters = xval_global_parameters.split('%')
    if len(xval_global_parameters) > 0:
        for param in xval_global_parameters:
            exec(param)
except:
    pass


#replace instance parameters with global_parameters
if len(global_parameters) > 0:
    for global_parameter in global_parameters:  #iterate global parameters
        flag = True
        global_param_name = global_parameter.split('=')[0]
        for e in experiment:  #iterate experiment instances
            local_parameters = experiment[e]
            for i in range(len(local_parameters)):  #iterate instance parameters
                instance_param_name = local_parameters[i].split('=')[0]
                if instance_param_name == global_param_name:
                    local_parameters[i] = global_parameter
                    flag = False
            #append parameter if not modified
            if flag == True:
                local_parameters.append(global_parameter)


experiment_folder = os.path.join(experiment_folder, dataset)
print ('Overwrite results: ' + str(overwrite_results))
print ('Debug mode: ' + str(debug_mode))

output_path = experiment_folder + '/experiment_' + str(num_experiment)
if not os.path.exists(output_path):
    os.makedirs(output_path)
description_path = output_path + '/experiment_' + str(num_experiment) + '_description.txt'

with open(description_path, "w") as text_file:
    text_file.write(short_description)

if isinstance(begin,list):
    iterable_experiments = begin
else:
    iterable_experiments = range(begin,end+1)

for num_run in iterable_experiments:
    results_name = output_path + '/results/results_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'
    results_name_BVL = output_path + '/results/BVL/results_BVL_' + dataset + '_exp' + str(num_experiment) + '_run' + str(num_run) + '.npy'

    temp_params = '/'.join(experiment[num_run])


    if overwrite_results:
        if debug_mode == False:
            try:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
            except:
                pass
        else:
            xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)

    else:  #if not overwrite results
        if not os.path.exists(results_name_BVL):  #not overwrite experiments
            if debug_mode == False:
                try:
                    xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
                except:
                    pass
            else:
                xval.run_experiment(num_experiment,num_run,num_folds,dataset,experiment_folder,temp_params, gpu_ID, task_type, generator)
        else:  #if result exists print below line
            print ('exp' + str(num_experiment) + ' run' + str(num_run) + ' already exists: skipping')
print('')
print ('REQUESTED EXPERIMENTS SUCCESSFULLY COMPLETED')
