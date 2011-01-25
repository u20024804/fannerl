-module(fann_example).

-export([run/0]).

run() ->
    Net = fann:create_standard({2,3,1}),
    fann:set_activation_function_hidden(Net, 5),
    fann:set_activation_function_hidden(Net, 5),
    fann:train_on_file(Net, "xor.data", 1000000, 0, 0.001),
    Mse = fann:get_mse(Net),
    fann:save(Net, "xor.net"),    
    fann:destroy(Net),
    Mse.
   
