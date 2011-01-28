-module(fann).

-export([create_standard/1, train_on_file/5, get_mse/1, save/2, destroy/1,
	 set_activation_function_hidden/2, set_activation_function_output/2,
	 get_activation_function/3, print_parameters/1, print_connections/1,
	 run/2, train_on_data/5]).

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

train_on_data(_,_,_,_,_) ->
    exit(nif_library_not_loaded).
