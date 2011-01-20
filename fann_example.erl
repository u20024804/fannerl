-module(fann_example).

-export([run/0]).

run() ->
    Net = fann:create_standard({2,3,1}),
    Net1= fann:set_activation_function_hidden(Net, 5),
    Net2= fann:set_activation_function_hidden(Net1, 5),
    Net3 = fann:train_on_file(Net2, "xor.data", 500000, 1000, 0.001),
    fann:save(Net3).
    %%fann:destroy(Net3).
    
