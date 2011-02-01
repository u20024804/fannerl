-module(fann).

-export([create_standard/1, train_on_file/5, get_mse/1, save/2, destroy/1,
	 set_activation_function_hidden/2, set_activation_function_output/2,
	 get_activation_function/3, print_parameters/1, print_connections/1,
	 run/2, test/3, randomize_weights/3, train_on_data/5, create_train/1,
	 shuffle_train_data/1, scale_train/2, descale_train/2, 
	 set_input_scaling_params/4, set_output_scaling_params/4,
	 set_scaling_params/6, clear_scaling_params/1, scale_input_train_data/3,
	 scale_output_train_data/3, scale_train_data/3, merge_train_data/2,
	 subset_train_data/3, num_input_train_data/1, num_output_train_data/1,
	 save_train/2, get_training_algorithm/1, set_training_algorithm/2,
	 get_learning_rate/1, set_learning_rate/2, get_learning_momentum/1,
	 set_learning_momentum/2]).

-on_load(init/0).

init() ->
    erlang:load_nif("./fann_nif", 0).

create_standard(_) ->
    exit(nif_library_not_loaded).

train_on_file(_,_,_,_,_) ->
    exit(nif_library_not_loaded).

get_mse(_) ->
    exit(nif_library_not_loaded).

save(_,_) ->
    exit(nif_library_not_loaded).

destroy(_) ->
    exit(nif_library_not_loaded).

set_activation_function_hidden(_,_) ->
    exit(nif_library_not_loaded).

set_activation_function_output(_,_) ->
    exit(nif_library_not_loaded).

get_activation_function(_,_,_) ->
    exit(nif_library_not_loaded).

print_parameters(_) ->
    exit(nif_library_not_loaded).

print_connections(_) ->
    exit(nif_library_not_loaded).

run(_,_) ->
    exit(nif_library_not_loaded).

test(_,_,_) ->
    exit(nif_library_not_loaded).

randomize_weights(_,_,_) ->
    exit(nif_library_not_loaded).

train_on_data(_,_,_,_,_) ->
    exit(nif_library_not_loaded).

create_train(_) ->
    exit(nif_library_not_loaded).

shuffle_train_data(_) ->
    exit(nif_library_not_loaded).
 
scale_train(_,_) ->
    exit(nif_library_not_loaded).
descale_train(_,_) ->

    exit(nif_library_not_loaded).
set_input_scaling_params(_,_,_,_) ->
    exit(nif_library_not_loaded).

set_output_scaling_params(_,_,_,_) ->
    exit(nif_library_not_loaded).

set_scaling_params(_,_,_,_,_,_) ->
    exit(nif_library_not_loaded).

clear_scaling_params(_) ->
    exit(nif_library_not_loaded).

scale_input_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

scale_output_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

scale_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

merge_train_data(_,_) ->
    exit(nif_library_not_loaded).

subset_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

num_input_train_data(_) ->
    exit(nif_library_not_loaded).

num_output_train_data(_) ->
    exit(nif_library_not_loaded).

save_train(_,_) ->
    exit(nif_library_not_loaded).

get_training_algorithm(_) ->
    exit(nif_library_not_loaded).

set_training_algorithm(_,_) ->
    exit(nif_library_not_loaded).

get_learning_rate(_) ->
    exit(nif_library_not_loaded).

set_learning_rate(_,_) ->
    exit(nif_library_not_loaded).

get_learning_momentum(_) ->
    exit(nif_library_not_loaded).

set_learning_momentum(_,_) ->
    exit(nif_library_not_loaded).
