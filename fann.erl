-module(fann).

-export([create_standard/1, train_on_file/5, get_mse/1]).

-on_load(init/0).

init() ->
    erlang:load_nif("./fann_nif", 0).

create_standard(_) ->
    exit(nif_library_not_loaded).

train_on_file(_,_,_,_,_) ->
    exit(nif_library_not_loaded).

get_mse(_) ->
    exit(nif_library_not_loaded).
