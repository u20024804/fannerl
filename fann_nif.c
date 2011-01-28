#include <erl_nif.h>
#include <fann.h>

static ErlNifResourceType * FANN_POINTER=NULL;
static ErlNifResourceType * TRAIN_DATA_THREAD=NULL;

struct fann_resource {
  struct fann * ann;
};

struct train_data_thread {
  ErlNifTid tid;
};

struct train_data_thread_data {
  struct fann_resource * resource;  
  struct fann_train_data * train_data;
  unsigned int max_epochs, epochs_between_reports;
  double desired_error;
  ErlNifPid to_pid;
};

static fann_type ** global_fann_array_inputs;
static fann_type ** global_fann_array_outputs;
static int global_array_set = 0;

static void * thread_run_fann_train_on_data(void *);

static void create_train_data(unsigned int num, unsigned int num_input,
			      unsigned int num_output, fann_type * input,
			      fann_type * output) {
  int i;
  for(i=0; i< num_input; ++i) {
    input[i] = global_fann_array_inputs[num][i];
  }
  for(i=0; i< num_output; ++i) {
    output[i] = global_fann_array_outputs[num][i];
  }
}

int check_and_convert_uint_array(ErlNifEnv* env, 
				 const ERL_NIF_TERM * tuple_array,
				 int tuple_size, 
				 unsigned int * converted_array) {
  int i;
  unsigned int array_value;
  ERL_NIF_TERM term;
  int * point;
  for(i = 0; i < tuple_size; ++i) {
    term = *(tuple_array + i);
    if(enif_get_uint(env, term, &array_value)) {
      point = converted_array + i;
      *point =  array_value;
    } else {
      return 0;
    }
  }
  return 1;
}

int check_and_convert_fann_type_array(ErlNifEnv* env, 
				      const ERL_NIF_TERM * tuple_array,
				      int tuple_size, 
				      fann_type * converted_array) {
  int i;
  double array_value;
  long long_array_value;
  ERL_NIF_TERM term;
  fann_type * point;
  for(i = 0; i < tuple_size; ++i) {
    term = *(tuple_array + i);
    if(enif_get_double(env, term, &array_value)) {
      point = converted_array + i;
      *point =  (fann_type)array_value;
    } else if(enif_get_long(env, term, &long_array_value)) {
      point = converted_array + i;
      *point =  (fann_type)long_array_value;
      
    } else {
      return 0;
    }
  }
  return 1;
}

void convert_to_erl_nif_array_from_fann_type(ErlNifEnv* env,
					    fann_type * fann_array,
					    ERL_NIF_TERM * tuple_array,
					    unsigned int size) {
  int i;
  fann_type array_value;
  ERL_NIF_TERM erl_double;
  for(i=0; i < size; ++i) {
    array_value = *(fann_array + i);
    erl_double = enif_make_double(env, (double)array_value);
    *(tuple_array + i) = erl_double;
  }
  return;
}

static void destroy_fann_pointer(ErlNifEnv * env, void * resource) {
  fann_destroy(((struct fann_resource *) resource)->ann);
}

static void destroy_train_data_thread(ErlNifEnv * env, void * resource) {
  enif_thread_join(((struct train_data_thread *)resource)->tid, NULL);
}

static int load(ErlNifEnv * env, void ** priv_data, ERL_NIF_TERM load_info){
  FANN_POINTER = enif_open_resource_type(env, 
					 NULL, 
					 "fann_pointer", 
					 destroy_fann_pointer, 
					 ERL_NIF_RT_CREATE |
					 ERL_NIF_RT_TAKEOVER,
					 NULL);
  TRAIN_DATA_THREAD = enif_open_resource_type(env, 
					      NULL, 
					      "train_data_thread", 
					      destroy_train_data_thread, 
					      ERL_NIF_RT_CREATE |
					      ERL_NIF_RT_TAKEOVER,
					      NULL);

  if(FANN_POINTER == NULL || TRAIN_DATA_THREAD == NULL) {
    return -1;
  } else {
    return 0;
  }
}
  

static ERL_NIF_TERM create_standard_nif(ErlNifEnv* env, int argc,
					const ERL_NIF_TERM argv[]) {
  
  int x, ret, tuple_size;
  const ERL_NIF_TERM * tuple_array;
  unsigned int * converted_array;
  struct fann_resource * resource;
  ERL_NIF_TERM result;
  resource = enif_alloc_resource(FANN_POINTER, sizeof(struct fann_resource));
  if(enif_get_tuple(env, argv[0], &tuple_size, &tuple_array)) {
    converted_array = malloc(tuple_size*sizeof(unsigned int));
    if(check_and_convert_uint_array(env, tuple_array, tuple_size, 
				    converted_array)) {
      resource->ann = fann_create_standard_array(tuple_size, 
				       converted_array);
      if(converted_array!=NULL) {
	free(converted_array);
	converted_array=NULL;
      }      
      result = enif_make_resource(env, (void *)resource);
      enif_release_resource(resource);
      return result;
    } else {
      if(converted_array!=NULL) {
	free(converted_array);
	converted_array=NULL;
      }
      enif_release_resource(resource);
      return enif_make_badarg(env);
    }
  } else {
    enif_release_resource(resource);
    return enif_make_badarg(env);
  }
}

static ERL_NIF_TERM train_on_file_nif(ErlNifEnv* env, int argc,
				      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int string_length, max_epochs, epochs_between_reports;
  char * file_name;
  double desired_error;
  ERL_NIF_TERM result;
  
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_list_length(env, argv[1], &string_length)){
    return enif_make_badarg(env);
  }
  file_name = malloc((string_length+1)*sizeof(char));
  enif_get_string(env, argv[1], file_name, string_length+1, ERL_NIF_LATIN1);
  if(!enif_get_uint(env, argv[2], &max_epochs)) {
    free(file_name);
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[3], &epochs_between_reports)) {
    free(file_name);
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[4], &desired_error)) {
    free(file_name);
    return enif_make_badarg(env);
  }
  fann_train_on_file(resource->ann, file_name, max_epochs, 
		     epochs_between_reports, desired_error);
  free(file_name);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_mse_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double mse;
  
  ERL_NIF_TERM result;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  mse = fann_get_MSE(resource->ann);
  result = enif_make_double(env, mse);
  return result;
}

static ERL_NIF_TERM save_nif(ErlNifEnv* env, int argc, 
			     const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  char * file_name;
  unsigned int string_length; 
  ERL_NIF_TERM result;
  
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_list_length(env, argv[1], &string_length)){
    return enif_make_badarg(env);
  }
  file_name = malloc((string_length+1)*sizeof(char));
  enif_get_string(env, argv[1], file_name, string_length+1, ERL_NIF_LATIN1);
  fann_save(resource->ann, file_name);
  free(file_name);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM destroy_nif(ErlNifEnv* env, int argc, 
			     const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  fann_destroy(resource->ann);
  enif_release_resource(resource);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_function_hidden_nif(ErlNifEnv* env, 
						       int argc, 
						       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int activation_function;
  ERL_NIF_TERM result;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[1], &activation_function)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_function_hidden(resource->ann, activation_function);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_function_output_nif(ErlNifEnv* env, 
						       int argc, 
						       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int activation_function;
  ERL_NIF_TERM result;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[1], &activation_function)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_function_output(resource->ann, activation_function);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_activation_function_nif(ErlNifEnv* env, 
						int argc, 
						const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int activation_function, layer, neuron;
  ERL_NIF_TERM result;
      
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[1], &layer)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[1], &neuron)) {
    return enif_make_badarg(env);
  }  
  activation_function = fann_get_activation_function(resource->ann, layer,
						     neuron);
  if(activation_function != -1) {
    return enif_make_atom(env, FANN_ACTIVATIONFUNC_NAMES[activation_function]);
  } else {
    return enif_make_atom(env, "neuron_does_not_exist");
  }
}

static ERL_NIF_TERM print_parameters_nif(ErlNifEnv* env, 
					 int argc, 
					 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  fann_print_parameters(resource->ann);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM print_connections_nif(ErlNifEnv* env, 
					  int argc, 
					  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  fann_print_connections(resource->ann);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM run_nif(ErlNifEnv* env, 
			    int argc, 
			    const ERL_NIF_TERM argv[]) {
  const ERL_NIF_TERM * tuple_array;
  ERL_NIF_TERM * output_tuple_array;
  ERL_NIF_TERM result;
  fann_type * converted_array, * output_array;
  struct fann_resource * resource;
  int tuple_size;
  unsigned int num_outputs;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(enif_get_tuple(env, argv[1], &tuple_size, &tuple_array)) {
    converted_array = malloc(tuple_size*sizeof(fann_type));
    if(check_and_convert_fann_type_array(env, tuple_array, tuple_size, 
					 converted_array)) {
      output_array = fann_run(resource->ann, converted_array);
      num_outputs = fann_get_num_output(resource->ann);
      output_tuple_array = malloc(num_outputs*sizeof(const ERL_NIF_TERM));
      
      convert_to_erl_nif_array_from_fann_type(env, output_array, 
					      output_tuple_array, num_outputs);
      result = enif_make_tuple_from_array(env, output_tuple_array, num_outputs);
      free(output_tuple_array);
      free(converted_array);
      return result;
    }
    free(converted_array);
    return enif_make_badarg(env);
  }
  return enif_make_badarg(env);
}

static ERL_NIF_TERM test_nif(ErlNifEnv* env, 
			    int argc, 
			    const ERL_NIF_TERM argv[]) {
  const ERL_NIF_TERM * tuple_array;
  fann_type * converted_input, * converted_output;
  struct fann_resource * resource;
  int tuple_size;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_tuple(env, argv[1], &tuple_size, &tuple_array)) {
    return enif_make_badarg(env);
  }
  converted_input = malloc(tuple_size*sizeof(fann_type));
  if(!check_and_convert_fann_type_array(env, tuple_array, tuple_size, 
				       converted_input)) {
    free(converted_input);
    return enif_make_badarg(env);
  }
  if(!enif_get_tuple(env, argv[2], &tuple_size, &tuple_array)) {
    free(converted_input);
    return enif_make_badarg(env);
  }
  converted_output = malloc(tuple_size*sizeof(fann_type));
  if(!check_and_convert_fann_type_array(env, tuple_array, tuple_size, 
				       converted_output)) {
    free(converted_input);
    free(converted_output);
    return enif_make_badarg(env);
  }
  fann_test(resource->ann, converted_input, converted_output);
  free(converted_input);
  free(converted_output);
  return enif_make_atom(env, "ok");  
}

static ERL_NIF_TERM randomize_weights_nif(ErlNifEnv* env, 
					  int argc, 
					  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double min, max;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &max)) {
    return enif_make_badarg(env);
  }
  fann_randomize_weights(resource->ann, (fann_type)min, (fann_type)max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM train_on_data_nif(ErlNifEnv* env, 
				      int argc, 
				      const ERL_NIF_TERM argv[]) {
  const ERL_NIF_TERM * tuple_array;
  ERL_NIF_TERM list, tail, element, temp_list, temp_tail, temp_head, 
    input_element, output_element;
  fann_type * converted_array, * output_array;
  fann_type ** fann_array_inputs, ** fann_array_outputs; 
  double desired_error;
  struct fann_resource * resource;
  int tuple_size, i, z;
  unsigned int list_length, set_list_length, num_inputs, num_outputs, 
    max_epochs, epochs_between_reports;
  struct fann_train_data * train_data;
  struct train_data_thread * thread_tid;
  struct train_data_thread_data * thread_data;
  ErlNifPid self;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  // First check that it is a list
  if(!enif_is_list(env, argv[1])) {
    return enif_make_badarg(env);
  }
  if(!enif_get_list_length(env, argv[1], &list_length)) {
    return enif_make_badarg(env);
  }
  num_inputs = fann_get_num_input(resource->ann);
  num_outputs = fann_get_num_output(resource->ann);
  fann_array_inputs = malloc(list_length*sizeof(fann_type *));
  fann_array_outputs = malloc(list_length*sizeof(fann_type *));
  list = argv[1];
  for(i=0; i < list_length; ++i) {
    if(!enif_get_list_cell(env, list, &element, &tail)) {
      free(fann_array_inputs);
      free(fann_array_outputs);      
      return enif_make_badarg(env);
    }
    if(!enif_is_list(env, element)) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return enif_make_badarg(env);
    }
    // get the size of the internal list that contains both a tuple
    // of inputs and a tuple of outputs
    if(!enif_get_list_length(env, element, &set_list_length)) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return enif_make_badarg(env);
    }
    if(set_list_length!=2) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return enif_make_badarg(env);
    }
    for(z=0; z < 2; ++z) {
      if(enif_get_list_cell(env, element, &temp_head, &temp_tail)) {
	  if(!enif_is_tuple(env, temp_head)) {
	    free(fann_array_inputs);
	    free(fann_array_outputs);
	    return enif_make_badarg(env);
	  }
	  if(!enif_get_tuple(env, temp_head, &tuple_size, &tuple_array)) {
	    free(fann_array_inputs);
	    free(fann_array_outputs);
	    return enif_make_badarg(env);
	  }
	  converted_array = malloc(tuple_size*sizeof(fann_type));
	  if(!check_and_convert_fann_type_array(env, tuple_array, tuple_size, 
						converted_array)) {
	    free(converted_array);
	    free(fann_array_inputs);
	    free(fann_array_outputs);
	    return enif_make_badarg(env);
	  }
	  if(z == 0) {
	    *(fann_array_inputs + i) = converted_array;
	  } else if(z == 1) {
	    *(fann_array_outputs + i) = converted_array;
	  }
	  element = temp_tail;
      } else {
	free(fann_array_inputs);
	free(fann_array_outputs);
	return enif_make_badarg(env);
      }     
      element = temp_tail;
    }
    list = tail;
  }  
  if(!enif_get_uint(env, argv[2], &max_epochs)) {
    free(fann_array_inputs);
    free(fann_array_outputs);
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[3], &epochs_between_reports)) {
    free(fann_array_inputs);
    free(fann_array_outputs);
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[4], &desired_error)) {
    free(fann_array_inputs);
    free(fann_array_outputs);
    return enif_make_badarg(env);
  }
  global_fann_array_inputs = fann_array_inputs;
  global_fann_array_outputs = fann_array_outputs;
  train_data = fann_create_train_from_callback(list_length, num_inputs, 
					       num_outputs,create_train_data);
  // get pid of self
  enif_self(env, &self);
  // Initalize thread_tid resource so that the thread will be joined
  // automatically by the GC
  thread_tid = enif_alloc_resource(TRAIN_DATA_THREAD, 
				   sizeof(struct train_data_thread));
  thread_data = malloc(sizeof(struct train_data_thread_data));
  // Initialize thread_data struct which will be sent to the thread
  thread_data->resource = resource;
  thread_data->train_data = train_data;
  thread_data->max_epochs = max_epochs;
  thread_data->epochs_between_reports = epochs_between_reports;
  thread_data->desired_error = desired_error;
  thread_data->to_pid = self;
  enif_thread_create("train_data_thread", &(thread_tid->tid), 
		     thread_run_fann_train_on_data, thread_data, NULL);
  free(fann_array_inputs);
  free(fann_array_outputs);
  return enif_make_atom(env, "ok");
}

static void * thread_run_fann_train_on_data(void * input_thread_data){
  ErlNifEnv * this_env;
  struct train_data_thread_data * thread_data;
  thread_data = (struct train_data_thread_data *)input_thread_data;
  fann_train_on_data(thread_data->resource->ann, 
		     thread_data->train_data, 
		     thread_data->max_epochs, 
		     thread_data->epochs_between_reports, 
		     thread_data->desired_error);
  fann_destroy_train(thread_data->train_data);
  this_env = enif_alloc_env();
  enif_send(NULL, &(thread_data->to_pid), this_env, 
	    enif_make_atom(this_env, "fann_train_on_data_complete"));
  free(thread_data);
  enif_free_env(this_env);
  enif_thread_exit(NULL);
}
      
static ErlNifFunc nif_funcs[] =
{
  {"create_standard", 1, create_standard_nif},
  {"train_on_file", 5, train_on_file_nif},
  {"get_mse", 1, get_mse_nif},
  {"save", 2, save_nif},
  {"destroy", 1, destroy_nif},
  {"set_activation_function_hidden", 2, set_activation_function_hidden_nif},
  {"set_activation_function_output", 2, set_activation_function_output_nif},
  {"get_activation_function", 3, get_activation_function_nif},
  {"print_parameters", 1, print_parameters_nif},
  {"print_connections", 1, print_connections_nif},
  {"run", 2, run_nif},
  {"test", 3, test_nif},
  {"randomize_weights", 3, randomize_weights_nif},
  {"train_on_data", 5, train_on_data_nif}
};

ERL_NIF_INIT(fann,nif_funcs,load,NULL,NULL,NULL)
