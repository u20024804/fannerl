/* fannerl (c) by Erik Axling

 fannerl is licensed under a
 Creative Commons Attribution-ShareAlike 3.0 Unported License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/3.0/>.
*/

#include <erl_nif.h>
#include <fann.h>
#include <string.h>
#include <ctype.h>

static ErlNifResourceType * FANN_POINTER=NULL;
static ErlNifResourceType * TRAIN_DATA_THREAD=NULL;
static ErlNifResourceType * TRAIN_DATA_RESOURCE=NULL;

struct fann_resource {
  struct fann * ann;
};

struct train_data_thread {
  ErlNifTid tid;
};

struct train_data_resource {
  struct fann_train_data * train_data;
  
};

struct train_data_thread_data {
  struct fann_resource * resource;  
  struct fann_train_data * train_data;
  unsigned int max_epochs, epochs_between_reports;
  double desired_error;
  ErlNifPid to_pid;
  ERL_NIF_TERM reference;
};

struct train_file_thread_data {
  struct fann_resource * resource;  
  char * file_name;
  unsigned int max_epochs, epochs_between_reports;
  double desired_error;
  ErlNifPid to_pid;
  ERL_NIF_TERM reference;
};

static fann_type ** global_fann_array_inputs;
static fann_type ** global_fann_array_outputs;

static void * thread_run_fann_train_on_data(void *);
static void * thread_run_fann_train_on_file(void *);
static int get_train_data_from_erl_input(ErlNifEnv *,
					 ERL_NIF_TERM, 
					 unsigned int *,
					 unsigned int *,
					 unsigned int *);
int get_activation_function(char *, int *);
int get_error_function(char *, int *);
int get_stop_function(char *, int *);
char * strtolower(const char *);

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
  unsigned int * point;
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
  printf("Ann pointer at destroy: %i\nResource pointer at destroy: %i\n", 
	 ((struct fann_resource *) resource)->ann, (int)resource);
  fann_destroy(((struct fann_resource *) resource)->ann);
}

static void destroy_train_data_thread(ErlNifEnv * env, void * resource) {
  enif_thread_join(((struct train_data_thread *)resource)->tid, NULL);
}
static void destroy_train_data_resource(ErlNifEnv * env, void * resource) {
  fann_destroy_train(((struct train_data_resource *)resource)->train_data);
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
  TRAIN_DATA_RESOURCE = enif_open_resource_type(env, 
						NULL, 
						"train_data_resource", 
						destroy_train_data_resource, 
						ERL_NIF_RT_CREATE |
						ERL_NIF_RT_TAKEOVER,
						NULL);
  
  if(FANN_POINTER == NULL || TRAIN_DATA_THREAD == NULL || 
     TRAIN_DATA_RESOURCE==NULL) {
    return -1;
  } else {
    return 0;
  }
}

static int reload(ErlNifEnv * env, void ** priv_data, ERL_NIF_TERM load_info) {
  printf("reload!\n");
  return 0;
}

static int upgrade(ErlNifEnv * env, void ** priv_data, void ** old_priv_data, 
		   ERL_NIF_TERM load_info) {
  printf("upgrade!\n");
  return 0;
}

static int unload(ErlNifEnv * env, void ** priv_data) {
  printf("unload\n");
  return 0;
}  

static ERL_NIF_TERM create_standard_nif(ErlNifEnv* env, int argc,
					const ERL_NIF_TERM argv[]) {
  
  int tuple_size;
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
      printf("ann pointer: %i\n", (int)resource->ann);
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
  ErlNifPid self;
  ERL_NIF_TERM reference;
  struct train_data_thread * thread_tid;
  struct train_file_thread_data * thread_data;
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
  // make unique reference
  reference = enif_make_ref(env);
  // get pid of self
  enif_self(env, &self);
  // Initalize thread_tid resource so that the thread will be joined
  // automatically by the GC
  thread_tid = enif_alloc_resource(TRAIN_DATA_THREAD, 
				   sizeof(struct train_data_thread));
  thread_data = malloc(sizeof(struct train_file_thread_data));
  // Initialize thread_data struct which will be sent to the thread
  thread_data->resource = resource;
  strcpy(thread_data->file_name, file_name); 
  free(file_name);
  thread_data->max_epochs = max_epochs;
  thread_data->epochs_between_reports = epochs_between_reports;
  thread_data->desired_error = desired_error;
  thread_data->to_pid = self;
  thread_data->reference = reference;
  enif_thread_create("train_file_thread", &(thread_tid->tid), 
		     thread_run_fann_train_on_file, thread_data, NULL);
  
  return enif_make_tuple2(env, enif_make_atom(env,"ok"), reference);
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
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_list_length(env, argv[1], &string_length)){
    return enif_make_badarg(env);
  }
  file_name = malloc((string_length+1)*sizeof(char));
  if(!enif_get_string(env,argv[1], file_name, string_length+1, ERL_NIF_LATIN1)){
    return enif_make_badarg(env);
  }
  fann_save(resource->ann, file_name);
  free(file_name);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_function_hidden_nif(ErlNifEnv* env, 
						       int argc, 
						       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int act_func; 
  unsigned int atom_length;
  char * activation_function;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  activation_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], activation_function, atom_length+1, 
		    ERL_NIF_LATIN1)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  if(!get_activation_function(activation_function, &act_func)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  free(activation_function);
  fann_set_activation_function_hidden(resource->ann, act_func);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_function_output_nif(ErlNifEnv* env, 
						       int argc, 
						       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int act_func;
  unsigned int atom_length;
  char * activation_function;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  activation_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], activation_function, atom_length+1, ERL_NIF_LATIN1)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  if(!get_activation_function(activation_function, &act_func)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  free(activation_function);
  fann_set_activation_function_output(resource->ann, act_func);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_activation_function_nif(ErlNifEnv* env, 
						int argc, 
						const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int length;
  int activation_function, layer, neuron;
  char * temp;
  ERL_NIF_TERM result;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[1], &layer)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[2], &neuron)) {
    return enif_make_badarg(env);
  }
  activation_function = fann_get_activation_function(resource->ann, layer,
						     neuron);
  if(activation_function != -1) {
    temp = strtolower(FANN_ACTIVATIONFUNC_NAMES[activation_function]);
    length = strlen(temp);
    result = enif_make_atom_len(env, temp, length);
    return result;
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
  double desired_error;
  struct fann_resource * resource;  
  unsigned int max_epochs, epochs_between_reports;
  struct train_data_thread * thread_tid;
  struct train_data_thread_data * thread_data;
  struct train_data_resource * train_data_resource;
  ErlNifPid self;
  ERL_NIF_TERM reference;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
    
  if(!enif_get_uint(env, argv[2], &max_epochs)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[3], &epochs_between_reports)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[4], &desired_error)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			  (void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  // make unique reference
  reference = enif_make_ref(env);
  // get pid of self
  enif_self(env, &self);
  // Initalize thread_tid resource so that the thread will be joined
  // automatically by the GC
  thread_tid = enif_alloc_resource(TRAIN_DATA_THREAD, 
				   sizeof(struct train_data_thread));
  thread_data = malloc(sizeof(struct train_data_thread_data));
  // Initialize thread_data struct which will be sent to the thread
  thread_data->resource = resource;
  thread_data->train_data = train_data_resource->train_data;
  thread_data->max_epochs = max_epochs;
  thread_data->epochs_between_reports = epochs_between_reports;
  thread_data->desired_error = desired_error;
  thread_data->to_pid = self;
  thread_data->reference = reference;
  enif_thread_create("train_data_thread", &(thread_tid->tid), 
		     thread_run_fann_train_on_data, thread_data, NULL);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), reference);
}

static ERL_NIF_TERM create_train_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_resource;
  struct fann_train_data * train_data;
  unsigned int train_length, num_inputs, num_outputs;
 
  ERL_NIF_TERM result;
  if(!get_train_data_from_erl_input(env, argv[0], &train_length, 
				    &num_inputs, &num_outputs)) {
    return enif_make_badarg(env);
  }  
  train_data = fann_create_train_from_callback(train_length, num_inputs, 
					       num_outputs,create_train_data);
  free(global_fann_array_inputs);
  free(global_fann_array_outputs);
  train_resource = enif_alloc_resource(TRAIN_DATA_RESOURCE, 
				       sizeof(struct train_data_resource));
  train_resource->train_data = train_data;
  result = enif_make_resource(env, train_resource);
  enif_release_resource(train_resource);
  return result;  
}

static ERL_NIF_TERM shuffle_train_data_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct train_data_resource * resource;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&resource)) {
    return enif_make_badarg(env);
  }
  fann_shuffle_train_data(resource->train_data);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM scale_train_nif(ErlNifEnv* env, 
				    int argc, 
				    const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  fann_scale_train(resource->ann, train_data_resource->train_data);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM descale_train_nif(ErlNifEnv* env, 
				      int argc, 
				      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  fann_descale_train(resource->ann, train_data_resource->train_data);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_input_scaling_params_nif(ErlNifEnv* env, 
					     int argc, 
					     const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  double input_min, input_max;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &input_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[3], &input_max)) {
    return enif_make_badarg(env);
  }
  fann_set_input_scaling_params(resource->ann, train_data_resource->train_data,
				(float)input_min, (float)input_max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_output_scaling_params_nif(ErlNifEnv* env, 
					      int argc, 
					      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  double output_min, output_max;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &output_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[3], &output_max)) {
    return enif_make_badarg(env);
  }
  fann_set_output_scaling_params(resource->ann, train_data_resource->train_data,
				(float)output_min, (float)output_max);
  return enif_make_atom(env, "ok");
}  

static ERL_NIF_TERM set_scaling_params_nif(ErlNifEnv* env, 
				       int argc, 
				       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  double input_min, input_max, output_min, output_max;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &input_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[3], &input_max)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[4], &output_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[5], &output_max)) {
    return enif_make_badarg(env);
  }
  fann_set_scaling_params(resource->ann, train_data_resource->train_data,
			  (float)input_min, (float)input_max,
			  (float)output_min, (float)output_max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM clear_scaling_params_nif(ErlNifEnv* env, 
					 int argc, 
					 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
    
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }  
  fann_clear_scaling_params(resource->ann);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM scale_input_train_data_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  double input_min, input_max;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &input_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &input_max)) {
    return enif_make_badarg(env);
  }
  fann_scale_input_train_data(train_data_resource->train_data,
			      (float)input_min, (float)input_max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM scale_output_train_data_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  double output_min, output_max;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &output_min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &output_max)) {
    return enif_make_badarg(env);
  }
  fann_scale_output_train_data(train_data_resource->train_data,
			      (float)output_min, (float)output_max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM scale_train_data_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  double min, max;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &min)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[2], &max)) {
    return enif_make_badarg(env);
  }
  fann_scale_train_data(train_data_resource->train_data,
			(float)min, (float)max);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM merge_train_data_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource1, * train_data_resource2, 
    * new_train_data_resource;
  ERL_NIF_TERM result;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource1)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_resource(env, argv[1], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource2)) {
    return enif_make_badarg(env);
  }
  new_train_data_resource = 
    enif_alloc_resource(TRAIN_DATA_RESOURCE,sizeof(struct train_data_resource));
  
  new_train_data_resource->train_data = 
    fann_merge_train_data(train_data_resource1->train_data,
			  train_data_resource2->train_data);
  result = enif_make_resource(env, new_train_data_resource);
  enif_release_resource(new_train_data_resource);
  return result;
}

static ERL_NIF_TERM subset_train_data_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource, * sub_train_data_resource;
  ERL_NIF_TERM result;
  unsigned int pos, length;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[1], &pos)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_uint(env, argv[2], &length)) {
    return enif_make_badarg(env);
  }
  sub_train_data_resource = 
    enif_alloc_resource(TRAIN_DATA_RESOURCE,sizeof(struct train_data_resource));
  
  sub_train_data_resource->train_data = 
    fann_subset_train_data(train_data_resource->train_data,
			   pos, length);
  result = enif_make_resource(env, sub_train_data_resource);
  enif_release_resource(sub_train_data_resource);
  return result;
}

static ERL_NIF_TERM length_train_data_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  unsigned int length;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }  
  length = fann_length_train_data(train_data_resource->train_data);
  return enif_make_uint(env, length);
}

static ERL_NIF_TERM num_input_train_data_nif(ErlNifEnv* env, 
					 int argc, 
					 const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  unsigned int num;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }  
  num = fann_num_input_train_data(train_data_resource->train_data);
  return enif_make_uint(env, num);
}

static ERL_NIF_TERM num_output_train_data_nif(ErlNifEnv* env, 
					  int argc, 
					  const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  unsigned int num;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }  
  num = fann_num_output_train_data(train_data_resource->train_data);
  return enif_make_uint(env, num);
}

static ERL_NIF_TERM save_train_nif(ErlNifEnv* env, int argc, 
			       const ERL_NIF_TERM argv[]) {
  struct train_data_resource * train_data_resource;
  char * file_name;
  unsigned int string_length;
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }  
  if(!enif_get_list_length(env, argv[1], &string_length)){
    return enif_make_badarg(env);
  }
  file_name = malloc((string_length+1)*sizeof(char));
  if(!enif_get_string(env,argv[1], file_name, string_length+1, ERL_NIF_LATIN1)){
    return enif_make_badarg(env);
  }
  fann_save_train(train_data_resource->train_data, file_name);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_training_algorithm_nif(ErlNifEnv* env, 
					       int argc, 
					       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int algo;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  algo = fann_get_training_algorithm(resource->ann);
  return enif_make_string(env, FANN_TRAIN_NAMES[algo], ERL_NIF_LATIN1);
}

static ERL_NIF_TERM set_training_algorithm_nif(ErlNifEnv* env, 
					       int argc, 
					       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int algo;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[1], &algo)) {
    return enif_make_badarg(env);
  }
  fann_set_training_algorithm(resource->ann, algo);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_learning_rate_nif(ErlNifEnv* env, 
					  int argc, 
					  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  float learning_rate;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  learning_rate = fann_get_learning_rate(resource->ann);
  return enif_make_double(env, learning_rate);
}

static ERL_NIF_TERM set_learning_rate_nif(ErlNifEnv* env, 
					  int argc, 
					  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double learning_rate ;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &learning_rate)) {
    return enif_make_badarg(env);
  }
  fann_set_learning_rate(resource->ann, (float)learning_rate);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_learning_momentum_nif(ErlNifEnv* env, 
					      int argc, 
					      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  float learning_momentum;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  learning_momentum = fann_get_learning_momentum(resource->ann);
  return enif_make_double(env, learning_momentum);
}

static ERL_NIF_TERM set_learning_momentum_nif(ErlNifEnv* env, 
					      int argc, 
					      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double learning_momentum ;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &learning_momentum)) {
    return enif_make_badarg(env);
  }
  fann_set_learning_momentum(resource->ann, (float)learning_momentum);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_function_nif(ErlNifEnv* env, 
						int argc, 
						const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  char * activation_function;
  int act_func, layer, neuron;
  unsigned int atom_length;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  activation_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], activation_function, atom_length+1, 
		    ERL_NIF_LATIN1)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  
  if(!get_activation_function(activation_function, &act_func)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  
  free(activation_function);
  if(!enif_get_int(env, argv[2], &layer)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[3], &neuron)) {
    return enif_make_badarg(env);
  }  
  fann_set_activation_function(resource->ann, act_func, layer, neuron);
  return enif_make_atom(env, "ok");
}
 
static ERL_NIF_TERM set_activation_function_layer_nif(ErlNifEnv* env, 
						      int argc, 
						      const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  char * activation_function;
  int act_func, layer;
  unsigned int atom_length;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  activation_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], activation_function, atom_length+1, 
		    ERL_NIF_LATIN1)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  if(!get_activation_function(activation_function, &act_func)) {
    free(activation_function);
    return enif_make_badarg(env);
  }
  free(activation_function);
  if(!enif_get_int(env, argv[2], &layer)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_function_layer(resource->ann, act_func, layer);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_activation_steepness_nif(ErlNifEnv* env, 
						 int argc, 
						 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int layer, neuron;
  fann_type activation_steepness;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[1], &layer)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[2], &neuron)) {
    return enif_make_badarg(env);
  }
  activation_steepness = fann_get_activation_steepness(resource->ann, layer, 
						       neuron);
  return enif_make_double(env, activation_steepness);
}

static ERL_NIF_TERM set_activation_steepness_nif(ErlNifEnv* env, 
						 int argc, 
						 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int layer, neuron;
  double activation_steepness;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &activation_steepness)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[2], &layer)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[3], &neuron)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_steepness(resource->ann, (fann_type)activation_steepness,
				layer, neuron);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_steepness_layer_nif(ErlNifEnv* env, 
						       int argc, 
						       const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int layer;
  double activation_steepness;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &activation_steepness)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_int(env, argv[2], &layer)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_steepness_layer(resource->ann, 
				      (fann_type)activation_steepness,
				      layer);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_steepness_hidden_nif(ErlNifEnv* env, 
							int argc, 
							const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double activation_steepness;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &activation_steepness)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_steepness_hidden(resource->ann, 
				       (fann_type)activation_steepness);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM set_activation_steepness_output_nif(ErlNifEnv* env, 
							int argc, 
							const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double activation_steepness;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &activation_steepness)) {
    return enif_make_badarg(env);
  }
  fann_set_activation_steepness_output(resource->ann, 
				       (fann_type)activation_steepness);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_train_error_function_nif(ErlNifEnv* env, 
						 int argc, 
						 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int train_error_func;
  char * temp;
  ERL_NIF_TERM result;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  train_error_func = fann_get_train_error_function(resource->ann);
  temp = strtolower(FANN_ERRORFUNC_NAMES[train_error_func]);
  result = enif_make_atom(env, temp);
  free(temp);
  return result;
}
static ERL_NIF_TERM set_train_error_function_nif(ErlNifEnv* env, 
						 int argc, 
						 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int atom_length;
  int train_error_func;
  char * error_function;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  error_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], error_function, atom_length+1, 
		    ERL_NIF_LATIN1)) {
    free(error_function);
    return enif_make_badarg(env);
  }
  if(!get_error_function(error_function, &train_error_func)) {
    free(error_function);
    return enif_make_badarg(env);
  }
  free(error_function);
  fann_set_train_error_function(resource->ann, train_error_func);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_train_stop_function_nif(ErlNifEnv* env, 
						int argc, 
						const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  int train_stop_func;
  char * temp=NULL;
  ERL_NIF_TERM result;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  train_stop_func = fann_get_train_stop_function(resource->ann);
  temp = strtolower(FANN_STOPFUNC_NAMES[train_stop_func]);
  result = enif_make_atom(env,temp);
  return result;
}
static ERL_NIF_TERM set_train_stop_function_nif(ErlNifEnv* env, 
						int argc, 
						const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int atom_length;
  int train_stop_func;
  char * stop_function;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_atom_length(env, argv[1], &atom_length, ERL_NIF_LATIN1)) {
    return enif_make_badarg(env);
  }
  stop_function = malloc((atom_length+1)*sizeof(char));
  if(!enif_get_atom(env, argv[1], stop_function, atom_length+1, 
		    ERL_NIF_LATIN1)) {
    free(stop_function);
    return enif_make_badarg(env);
  }
  if(!get_stop_function(stop_function, &train_stop_func)) {
    free(stop_function);
    return enif_make_badarg(env);
  }
  free(stop_function);
  fann_set_train_stop_function(resource->ann, train_stop_func);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_bit_fail_limit_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  fann_type bit_fail;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  bit_fail = fann_get_bit_fail_limit(resource->ann);
  return enif_make_double(env, bit_fail);
}

static ERL_NIF_TERM set_bit_fail_limit_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double bit_fail;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &bit_fail)) {
    return enif_make_badarg(env);
  }
  fann_set_bit_fail_limit(resource->ann, (fann_type)bit_fail);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_quickprop_mu_nif(ErlNifEnv* env, 
					 int argc, 
					 const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double quickprop_mu;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  quickprop_mu = fann_get_quickprop_mu(resource->ann);
  return enif_make_double(env, quickprop_mu);
}

static ERL_NIF_TERM set_quickprop_mu_nif(ErlNifEnv* env, 
					   int argc, 
					   const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double quickprop_mu;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &quickprop_mu)) {
    return enif_make_badarg(env);
  }
  fann_set_quickprop_mu(resource->ann, (float)quickprop_mu);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_rprop_increase_factor_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_increase;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  rprop_increase = fann_get_rprop_increase_factor(resource->ann);
  return enif_make_double(env, rprop_increase);
}

static ERL_NIF_TERM set_rprop_increase_factor_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_increase;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &rprop_increase)) {
    return enif_make_badarg(env);
  }
  fann_set_rprop_increase_factor(resource->ann, (float)rprop_increase);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_rprop_decrease_factor_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_decrease;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  rprop_decrease = fann_get_rprop_decrease_factor(resource->ann);
  return enif_make_double(env, rprop_decrease);
}

static ERL_NIF_TERM set_rprop_decrease_factor_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_decrease;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &rprop_decrease)) {
    return enif_make_badarg(env);
  }
  fann_set_rprop_decrease_factor(resource->ann, (float)rprop_decrease);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_rprop_delta_min_nif(ErlNifEnv* env, 
					    int argc, 
					    const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  rprop_delta = fann_get_rprop_delta_min(resource->ann);
  return enif_make_double(env, rprop_delta);
}

static ERL_NIF_TERM set_rprop_delta_min_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &rprop_delta)) {
    return enif_make_badarg(env);
  }
  fann_set_rprop_delta_min(resource->ann, (float)rprop_delta);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_rprop_delta_max_nif(ErlNifEnv* env, 
					    int argc, 
					    const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  rprop_delta = fann_get_rprop_delta_max(resource->ann);
  return enif_make_double(env, rprop_delta);
}

static ERL_NIF_TERM set_rprop_delta_max_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &rprop_delta)) {
    return enif_make_badarg(env);
  }
  fann_set_rprop_delta_max(resource->ann, (float)rprop_delta);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_rprop_delta_zero_nif(ErlNifEnv* env, 
					     int argc, 
					     const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  rprop_delta = fann_get_rprop_delta_zero(resource->ann);
  return enif_make_double(env, rprop_delta);
}

static ERL_NIF_TERM set_rprop_delta_zero_nif(ErlNifEnv* env, 
						  int argc, 
						  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  double rprop_delta;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_double(env, argv[1], &rprop_delta)) {
    return enif_make_badarg(env);
  }
  fann_set_rprop_delta_zero(resource->ann, (float)rprop_delta);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM get_bit_fail_nif(ErlNifEnv* env, 
				     int argc, 
				     const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  unsigned int bit_fail;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  bit_fail = fann_get_bit_fail(resource->ann);
  return enif_make_uint(env, bit_fail);
}

static ERL_NIF_TERM reset_mse_nif(ErlNifEnv* env, 
				  int argc, 
				  const ERL_NIF_TERM argv[]) {
  struct fann_resource * resource;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  fann_reset_MSE(resource->ann);
  return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM train_epoch_nif(ErlNifEnv* env,
				    int argc,
				    const ERL_NIF_TERM argv[]) {
  // Need to consider if this should be asynchronous
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  float mse;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE,
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  mse = fann_train_epoch(resource->ann, train_data_resource->train_data);
  return enif_make_double(env, mse);
}

static ERL_NIF_TERM test_data_nif(ErlNifEnv* env,
				  int argc, 
				  const ERL_NIF_TERM argv[]) {
  // Need to consider if this should be asynchronous
  struct fann_resource * resource;
  struct train_data_resource * train_data_resource;
  float mse;
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&resource)) {
    return enif_make_badarg(env);
  }
  if(!enif_get_resource(env, argv[0], TRAIN_DATA_RESOURCE, 
			(void **)&train_data_resource)) {
    return enif_make_badarg(env);
  }
  mse = fann_test_data(resource->ann, train_data_resource->train_data);
  return enif_make_double(env, mse);
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
  this_env = enif_alloc_env();
  enif_send(NULL, &(thread_data->to_pid), this_env,
	    enif_make_tuple2(this_env,
			     thread_data->reference,
			     enif_make_atom(this_env, 
					    "fann_train_on_data_complete")));
  free(thread_data);
  enif_free_env(this_env);
  enif_thread_exit(NULL); 
}

static void * thread_run_fann_train_on_file(void * input_thread_data){
  ErlNifEnv * this_env;
  struct train_file_thread_data * thread_data;
  thread_data = (struct train_file_thread_data *)input_thread_data;
  fann_train_on_file(thread_data->resource->ann, 
		     thread_data->file_name, 
		     thread_data->max_epochs, 
		     thread_data->epochs_between_reports, 
		     thread_data->desired_error);
  this_env = enif_alloc_env();
  enif_send(NULL, &(thread_data->to_pid), this_env, 
	    enif_make_tuple2(this_env,
			     thread_data->reference,
			     enif_make_atom(this_env, 
					    "fann_train_on_file_complete")));
  free(thread_data);
  enif_free_env(this_env);
  enif_thread_exit(NULL); 
}

static int get_train_data_from_erl_input(ErlNifEnv * env,
					 ERL_NIF_TERM data, 
					 unsigned int * train_length,
					 unsigned int * train_input,
					 unsigned int * train_output)
{
  fann_type ** fann_array_inputs, ** fann_array_outputs;
  const ERL_NIF_TERM * tuple_array;
  ERL_NIF_TERM list, tail, element, temp_tail, temp_head;
  unsigned int list_length, set_list_length;
  int tuple_size, i, z;
  fann_type * converted_array;
  // First check that it is a list
  if(!enif_is_list(env, data)) {
    return 0;
  }
  if(!enif_get_list_length(env, data, &list_length)) {
    return 0;
  }
  *(train_length)=list_length;
  fann_array_inputs = malloc(list_length*sizeof(fann_type *));
  fann_array_outputs = malloc(list_length*sizeof(fann_type *));
  list = data;
  for(i=0; i < list_length; ++i) {
    if(!enif_get_list_cell(env, list, &element, &tail)) {
      free(fann_array_inputs);
      free(fann_array_outputs);      
      return 0;
    }
    if(!enif_is_list(env, element)) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return 0;
    }
    // get the size of the internal list that contains both a tuple
    // of inputs and a tuple of outputs
    if(!enif_get_list_length(env, element, &set_list_length)) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return 0;
    }
    if(set_list_length!=2) {
      free(fann_array_inputs);
      free(fann_array_outputs);
      return 0;
    }
    for(z=0; z < 2; ++z) {
      if(enif_get_list_cell(env, element, &temp_head, &temp_tail)) {
	if(!enif_is_tuple(env, temp_head)) {
	  free(fann_array_inputs);
	  free(fann_array_outputs);
	  return 0;
	}
	if(!enif_get_tuple(env, temp_head, &tuple_size, &tuple_array)) {
	  free(fann_array_inputs);
	  free(fann_array_outputs);
	  return 0;
	}
	converted_array = malloc(tuple_size*sizeof(fann_type));
	if(!check_and_convert_fann_type_array(env, tuple_array, tuple_size, 
					      converted_array)) {
	  free(converted_array);
	  free(fann_array_inputs);
	  free(fann_array_outputs);
	  return 0;
	}
	if(z == 0) {
	  *train_input=tuple_size;
	  *(fann_array_inputs + i) = converted_array;
	} else if(z == 1) {
	  *train_output=tuple_size;
	  *(fann_array_outputs + i) = converted_array;
	}
	element = temp_tail;
      } else {
	free(fann_array_inputs);
	free(fann_array_outputs);
	return 0;
      }     
      element = temp_tail;
    }
    list = tail;
  }
  global_fann_array_inputs = fann_array_inputs;
  global_fann_array_outputs = fann_array_outputs;
  return 1;
}

int get_activation_function(char * activation_function, int * act_func) {
  if(strcmp(activation_function,"fann_linear")==0) {
    *act_func=0;
    return 1;
  } else if(strcmp(activation_function,"fann_threshold")==0) {
    *act_func=1;
    return 1;
  } else if(strcmp(activation_function,"fann_threshold_symmetric")==0) {
    *act_func=2;
    return 1;
  } else if(strcmp(activation_function,"fann_sigmoid")==0) {
    *act_func=3;
    return 1;
  } else if(strcmp(activation_function,"fann_sigmoid_stepwise")==0) {
    *act_func=4;
    return 1;
  } else if(strcmp(activation_function,"fann_sigmoid_symmetric")==0) {
    *act_func=5;
    return 1;
  } else if(strcmp(activation_function,"fann_gaussian")==0) {
    *act_func=6;
    return 1;
  } else if(strcmp(activation_function,"fann_gaussian_symmetric")==0) {
    *act_func=7;
    return 1;
  } else if(strcmp(activation_function,"fann_elliot")==0) {
    *act_func=8;
    return 1;
  } else if(strcmp(activation_function,"fann_elliot_symmetric")==0) {
    *act_func=9;
    return 1;
  } else if(strcmp(activation_function,"fann_linear_piece")==0) {
    *act_func=10;
    return 1;
  } else if(strcmp(activation_function,"fann_linear_piece_symmetric")==0) {
    *act_func=11;
    return 1;
  } else if(strcmp(activation_function,"fann_sin_symmetric")==0) {
    *act_func=12;
    return 1;
  } else if(strcmp(activation_function,"fann_cos_symmetric")==0) {
    *act_func=13;
    return 1;
  } else if(strcmp(activation_function,"fann_sin")==0) {
    *act_func=14;
    return 1;
  } else if(strcmp(activation_function,"fann_cos")==0) {
    *act_func=15;
    return 1;
  } else {
    return 0;
  }
}

int get_error_function(char * error_function, int * err_func) {
  if(strcmp(error_function,"fann_errorfunc_linear")==0) {
    *err_func=0;
    return 1;
  } else if(strcmp(error_function,"fann_errorfunc_tanh")==0) {
    *err_func=1;
    return 1;
  } else {
    return 0;
  }
}

int get_stop_function(char * stop_function, int * stop_func) {
  if(strcmp(stop_function,"fann_stopfunc_mse")==0) {
    *stop_func=0;
    return 1;
  } else if(strcmp(stop_function,"fann_stopfunc_bit")==0) {
    *stop_func=1;
    return 1;
  } else {
    return 0;
  }
}

char * strtolower(const char * string) {
  int i;
  size_t length;
  char * temp=NULL;
  length = strlen(string);
  temp=malloc(length*sizeof(char));
  strcpy(temp, string);
  for(i=0; i < length; ++i) {
    temp[i] = tolower(temp[i]);
  }
  return temp;
}
      
static ErlNifFunc nif_funcs[] =
{
  {"create_standard", 1, create_standard_nif},
  {"train_on_file", 5, train_on_file_nif},
  {"get_mse", 1, get_mse_nif},
  {"save", 2, save_nif},
  {"set_activation_function_hidden", 2, set_activation_function_hidden_nif},
  {"set_activation_function_output", 2, set_activation_function_output_nif},
  {"get_activation_function", 3, get_activation_function_nif},
  {"set_activation_function", 4, set_activation_function_nif},
  {"print_parameters", 1, print_parameters_nif},
  {"print_connections", 1, print_connections_nif},
  {"run", 2, run_nif},
  {"test", 3, test_nif},
  {"randomize_weights", 3, randomize_weights_nif},
  {"train_on_data", 5, train_on_data_nif},
  {"create_train", 1, create_train_nif},
  {"shuffle_train_data", 1, shuffle_train_data_nif},
  {"scale_train", 2, scale_train_nif},
  {"descale_train", 2, descale_train_nif},
  {"set_input_scaling_params", 4, set_input_scaling_params_nif},
  {"set_output_scaling_params", 4, set_output_scaling_params_nif},
  {"set_scaling_params", 6, set_scaling_params_nif},
  {"clear_scaling_params", 1, clear_scaling_params_nif},
  {"scale_input_train_data", 3, scale_input_train_data_nif},
  {"scale_output_train_data", 3, scale_output_train_data_nif},
  {"scale_train_data", 3, scale_train_data_nif},
  {"merge_train_data", 2, merge_train_data_nif},
  {"subset_train_data", 3, subset_train_data_nif},
  {"num_input_train_data", 1, num_input_train_data_nif},
  {"num_output_train_data", 1, num_output_train_data_nif},
  {"save_train", 2, save_train_nif},
  {"get_training_algorithm", 1, get_training_algorithm_nif},
  {"set_training_algorithm", 2, set_training_algorithm_nif},
  {"get_learning_rate", 1, get_learning_rate_nif},
  {"set_learning_rate", 2, set_learning_rate_nif},
  {"get_learning_momentum", 1, get_learning_momentum_nif},
  {"set_learning_momentum", 2, set_learning_momentum_nif},
  {"length_train_data", 1, length_train_data_nif},
  {"set_activation_function_layer", 3, set_activation_function_layer_nif},
  {"get_activation_steepness", 3, get_activation_steepness_nif},
  {"set_activation_steepness", 4, set_activation_steepness_nif},
  {"set_activation_steepness_layer", 3, set_activation_steepness_layer_nif},
  {"set_activation_steepness_hidden", 2, set_activation_steepness_hidden_nif},
  {"set_activation_steepness_output", 2, set_activation_steepness_output_nif},
  {"get_train_error_function", 1, get_train_error_function_nif},
  {"set_train_error_function", 2, set_train_error_function_nif},
  {"get_train_stop_function", 1, get_train_stop_function_nif},
  {"set_train_stop_function", 2, set_train_stop_function_nif},
  {"get_bit_fail_limit", 1, get_bit_fail_limit_nif},
  {"set_bit_fail_limit", 2, set_bit_fail_limit_nif},
  {"get_quickprop_mu", 1, get_quickprop_mu_nif},
  {"set_quickprop_mu", 2, set_quickprop_mu_nif},
  {"get_rprop_increase_factor", 1, get_rprop_increase_factor_nif},
  {"set_rprop_increase_factor", 2, set_rprop_increase_factor_nif},
  {"get_rprop_decrease_factor", 1, get_rprop_decrease_factor_nif},
  {"set_rprop_decrease_factor", 2, set_rprop_decrease_factor_nif},
  {"get_rprop_delta_min", 1, get_rprop_delta_min_nif},
  {"set_rprop_delta_min", 2, set_rprop_delta_min_nif},
  {"get_rprop_delta_max", 1, get_rprop_delta_max_nif},
  {"set_rprop_delta_max", 2, set_rprop_delta_max_nif},
  {"get_rprop_delta_zero", 1, get_rprop_delta_zero_nif},
  {"set_rprop_delta_zero", 2, set_rprop_delta_zero_nif},
  {"get_bit_fail", 1, get_bit_fail_nif},
  {"reset_mse", 1, reset_mse_nif},
  {"train_epoch", 2, train_epoch_nif},
  {"test_data", 2, test_data_nif},
};

ERL_NIF_INIT(fannerl,nif_funcs,load,reload,upgrade,unload)
