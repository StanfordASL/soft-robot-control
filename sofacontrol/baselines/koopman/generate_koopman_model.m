% This provides interface to open-source code for Koopman data-driven 
% model-based control. It can be used to build a lifted linear model

% Setup: Add this file to the base folder of following cloned github repo:
% https://github.com/ramvasudevan/soft-robot-koopman
% Save training and validation data in baselines/[robot_name]_koopman.py


% Goes through full process of getting a linear model from data
% 1. Gather training and validation data and save in req format
% 2. Load saved data and train this data
% 3. Validate data
% 4. Save data to as .mat file with functionality for interface with python

%% parameters to consider for tuning
Ts = 0.05; % sampling time to consider
obs_degree = 2;  % Degree of moniomials to consider
delay = 1;  % Nbr of delays to consider in observables
lasso = [10]; % Lasso parameter value



%% gather training data (need to prepare data file before running this)

% load in data file(s)
[ datafile_name , datafile_path ] = uigetfile( 'datafiles/koopman_paper_import/*.mat' ,...
    'Choose training data file...' );

[ valfile_name, valfile_path ] = uigetfile('datafiles/koopman_paper_import/*.mat' ,...
    'Choose validation data file...' );

training_data = load([datafile_path, datafile_name]);
% training_data.u = training_data.u(1:end-1,:);
training_data = data.resample(training_data, Ts);

val_data = load([valfile_path, valfile_name]);
% val_data.u = val_data.u(1:end-1,:);
val_data = data.resample(val_data, Ts);
data_inst = data();

data_inst.get_data4sysid(training_data, val_data,...
    'True', 'ee_pos_20Hz');


%%
[ datafile_name , datafile_path ] = uigetfile( 'datafiles/*.mat' , 'Choose data file for sysid...' );
data4sysid = load( [datafile_path , datafile_name] );

% data4sysid = load( [datafile_path , datafile_name] );


%% construct sysid class
ksysid_inst = ksysid( data4sysid, ...
                'model_type' , 'linear' ,...    % model type (linear or nonlinear)
                'obs_type' , { 'poly' } ,...    % type of basis functions
                'obs_degree' , [ obs_degree ] ,...       % "degree" of basis functions
                'snapshots' , Inf ,...          % Number of snapshot pairs
                'lasso' , lasso ,...           % L1 regularization term
                'delays' , delay );                 % Numer of state/input delays


%% train model(s)
models = ksysid_inst.train_models;


%% validate model(s)
% could also manually do this for one model at a time

results = cell( size(models.candidates) );    % store results in a cell array
err = cell( size(models.candidates) );    % store error in a cell array

if iscell(models.candidates)
    for i = 1 : length(models.candidates)
        [ results{i} , err{i} ] = models.valNplot_model( i );
    end
else
    [ results{1} , err{1} ] = models.valNplot_model;
end

%% If not saved whilst validating, you can opt to save it separately here
models.save_class( )
    
%% Save the aforementioned model (or select one if none is defined)
[ model_name , model_path] = uigetfile( 'systems/fromData/*.mat' ,...
    'Choose model to save to python...' );
model = load([model_path, model_name]);

export_to_python(model);


%% save model(s)



%% save 
% You do this based on the validation results.
% Call this function:



function export_to_python(model)
    % model should be of ksysid type
    py_data = struct();
    lin_model = struct();
    lin_model.A = model.sysid_class.model.A;
    lin_model.B = model.sysid_class.model.B;
    lin_model.C = model.sysid_class.model.C;
    lin_model.M = model.sysid_class.model.M;
    lin_model.K = model.sysid_class.model.K;
    py_data.model = lin_model;
    
    params = struct();
    params.n = model.sysid_class.params.n;
    params.m = model.sysid_class.params.m;
    params.N = model.sysid_class.params.N;
    params.nzeta = model.sysid_class.params.nzeta;
    params.Ts = model.sysid_class.params.Ts;
    params.scale = model.sysid_class.params.scale;
    params.delays = model.sysid_class.delays;
    params.obs_type = model.sysid_class.obs_type;
    params.obs_degree = model.sysid_class.obs_degree;
    
    py_data.params = params;
    save('py_model.mat', 'py_data', '-v7');

end