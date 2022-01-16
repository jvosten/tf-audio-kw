#' @title Generator preparation.
#' @description In this step we will list all audio .wav files into a tibble with 3 columns:
#' * `fname`: the file name;
#' * `class`: the label for each audio file;
#' * `class_id`: a unique integer number starting from zero for each class - used to one-hot encode the classes.
#' This will be useful to the next step when we will create a generator using the tfdatasets package.
#' @return A `tibble`
#' @param data_path Path to Speech Commands data dir.
#' @examples
#' library(tidyverse)
#' prep_tbl("data/speech_commands_v0.01/")
prep_tbl <- function(path) {
  files <- fs::dir_ls(
    path = path,
    recurse = TRUE,
    glob = "*.wav"
  )

  files <- files[!str_detect(files, "background_noise")]

  df <- tibble(
    fname = files,
    class = fname %>% str_extract("1/.*/") %>%
      str_replace_all("1/", "") %>%
      str_replace_all("/", ""),
    class_id = class %>% as.factor() %>% as.integer() - 1L
  )
}

#' @title Dataset generator
#' @description define a function called data_generator that will create the
#' generator depending on specified inputs.
#' @return An `environment` that definies how the TensorFlow graph should read
#' and pre-process data.
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @param batch_size
#' @param window_size_ms
#' @param window_stride_ms
#' @param shuffle
#' @examples
#' library(tfdatasets)
data_generator <- function(df,
                           batch_size,
                           window_size_ms,
                           window_stride_ms,
                           shuffle = TRUE) {

  # assume sampling rate is the same in all samples
  sampling_rate <-
    tf$audio$decode_wav(tf$io$read_file(tf$reshape(df$fname[[1]], list()))) %>% .$sample_rate

  samples_per_window <- (sampling_rate * window_size_ms) %/% 1000L
  stride_samples <-  (sampling_rate * window_stride_ms) %/% 1000L

  n_periods <-
    tf$shape(
      tf$range(
        samples_per_window %/% 2L,
        16000L - samples_per_window %/% 2L,
        stride_samples
      )
    )[1] + 1L

  n_fft_coefs <-
    (2 ^ tf$math$ceil(tf$math$log(
      tf$cast(samples_per_window, tf$float32)
    ) / tf$math$log(2)) /
      2 + 1L) %>% tf$cast(tf$int32)

  ds <- tensor_slices_dataset(df)

  if (shuffle == TRUE){
    ds <- ds %>%
      dataset_shuffle(buffer_size = 100)
  }

  ds <- ds %>%
    dataset_map(function(obs) {
      wav <-
        tf$audio$decode_wav(tf$io$read_file(tf$reshape(obs$fname, list())))
      samples <- wav$audio
      samples <- samples %>% tf$transpose(perm = c(1L, 0L))

      stft_out <- tf$signal$stft(samples,
                                 frame_length = samples_per_window,
                                 frame_step = stride_samples)

      magnitude_spectrograms <- tf$abs(stft_out)
      log_magnitude_spectrograms <- tf$math$log(magnitude_spectrograms + 1e-6)

      response <- tf$one_hot(obs$class_id, 30L)

      input <- tf$transpose(log_magnitude_spectrograms, perm = c(1L, 2L, 0L))
      list(input, response)
    })

  ds <- ds %>%
    dataset_repeat()

  ds <- ds %>%
    dataset_padded_batch(
      batch_size = batch_size,
      padded_shapes = list(
        tf$stack(list(
          n_periods,
          n_fft_coefs,-1L)),
        tf$constant(-1L,
                    shape = shape(1L))),
      drop_remainder = TRUE
    )

  ds
}

#' @title Define a Keras model.
#' @description Define a Keras model for the customer churn data.
#' @return A Keras model. Not compiled or run yet.
#' @param churn_recipe A `recipe` object from [prepare_recipe()].
#' @param units1 Number of neurons in the first layer.
#' @param units2 Number of neurons in the second layer.
#' @param act1 Activation function for layer 1.
#' @param act2 Activation function for layer 2.
#' @param act3 Activation function for layer 3.
#' @examples
#' library(keras)
#' library(recipes)
#' library(rsample)
#' library(tidyverse)
#' data <- split_data("data/customer_churn.csv")
#' recipe <- prepare_recipe(data)
#' model <- define_model(recipe, 16, 16, "relu", "relu", "sigmoid")
#' model
define_model <- function(churn_recipe, units1, units2, act1, act2, act3) {
  input_shape <- ncol(
    juice(churn_recipe, all_predictors(), composition = "matrix")
  )
  keras_model_sequential() %>%
    layer_dense(
      units = units1,
      kernel_initializer = "uniform",
      activation = act1,
      input_shape = input_shape
    ) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(
      units = units2,
      kernel_initializer = "uniform",
      activation = act2
    ) %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(
      units = 1,
      kernel_initializer = "uniform",
      activation = act3
    )
}

#' @title Define, compile, and train a Keras model.
#' @description Define, compile, and train a Keras model on the training
#'   dataset.
#' @details The first time you run Keras in an R session,
#'   TensorFlow usually prints verbose ouput such as
#'   "Your CPU supports instructions that this TensorFlow binary was not compiled to use:"
#'   and "OMP: Info #171: KMP_AFFINITY:". You can safely ignore these messages.
#' @return A trained Keras model.
#' @inheritParams define_model
#' @examples
#' library(keras)
#' library(recipes)
#' library(rsample)
#' library(tidyverse)
#' data <- split_data("data/customer_churn.csv")
#' recipe <- prepare_recipe(data)
#' train_model(recipe, 16, 16, "relu", "relu", "sigmoid")
train_model <- function(
  churn_recipe,
  units1 = 16,
  units2 = 16,
  act1 = "relu",
  act2 = "relu",
  act3 = "sigmoid"
) {
  model <- define_model(churn_recipe, units1, units2, act1, act2, act3)
  compile(
    model,
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  x_train_tbl <- juice(
    churn_recipe,
    all_predictors(),
    composition = "matrix"
  )
  y_train_vec <- juice(churn_recipe, all_outcomes()) %>%
    pull()
  fit(
    object = model,
    x = x_train_tbl,
    y = y_train_vec,
    batch_size = 32,
    epochs = 32,
    validation_split = 0.3,
    verbose = 0
  )
  model
}

#' @title Test accuracy.
#' @description Compute the classification accuracy of a trained Keras model
#'   on the test dataset.
#' @return Classification accuracy of a trained Keras model on the test
#'   dataset.
#' @inheritParams define_model
#' @examples
#' library(keras)
#' library(recipes)
#' library(rsample)
#' library(tidyverse)
#' library(yardstick)
#' data <- split_data("data/customer_churn.csv")
#' recipe <- prepare_recipe(data)
#' model <- train_model(recipe, 16, 16, "relu", "relu", "sigmoid")
#' test_accuracy(data, recipe, model)
test_accuracy <- function(churn_data, churn_recipe, model) {
  testing_data <- bake(churn_recipe, testing(churn_data))
  x_test_tbl <- testing_data %>%
    select(-Churn) %>%
    as.matrix()
  y_test_vec <- testing_data %>%
    select(Churn) %>%
    pull()
  yhat_keras_class_vec <- model %>%
    predict(x_test_tbl) %>%
    `>`(0.5) %>%
    as.integer() %>%
    as.factor() %>%
    fct_recode(yes = "1", no = "0")
  yhat_keras_prob_vec <-
    model %>%
    predict(x_test_tbl) %>%
    as.vector()
  test_truth <- y_test_vec %>%
    as.factor() %>%
    fct_recode(yes = "1", no = "0")
  estimates_keras_tbl <- tibble(
    truth = test_truth,
    estimate = yhat_keras_class_vec,
    class_prob = yhat_keras_prob_vec
  )
  estimates_keras_tbl %>%
    conf_mat(truth, estimate) %>%
    summary() %>%
    filter(.metric == "accuracy") %>%
    pull(.estimate)
}

#' @title Benchmark a Keras model.
#' @description Define, compile, and train a Keras model on the training
#'   dataset. Then, benchmark it on the test dataset and return summaries.
#' @details The first time you run Keras in an R session,
#'   TensorFlow usually prints verbose ouput such as
#'   "Your CPU supports instructions that this TensorFlow binary was not compiled to use:"
#'   and "OMP: Info #171: KMP_AFFINITY:". You can safely ignore these messages.
#' @return A data frame with one row and the following columns:
#'   * `accuracy`: classification accuracy on the test dataset.
#'   * `units1`: number of neurons in layer 1.
#'   * `units2`: number of neurons in layer 2.
#'   * `act1`: number of neurons in layer 1.
#'   * `act2`: number of neurons in layer 2.
#'   * `act3`: number of neurons in layer 3.
#' @inheritParams define_model
#' @inheritParams test_accuracy
#' @examples
#' library(keras)
#' library(recipes)
#' library(rsample)
#' library(tidyverse)
#' data <- split_data("data/customer_churn.csv")
#' recipe <- prepare_recipe(data)
#' test_model(data, recipe, 16, 16, "relu", "relu", "sigmoid")
test_model <- function(
  churn_data,
  churn_recipe,
  units1 = 16,
  units2 = 16,
  act1 = "relu",
  act2 = "relu",
  act3 = "sigmoid"
) {
  model <- train_model(churn_recipe, units1, units2, act1, act2, act3)
  accuracy <- test_accuracy(churn_data, churn_recipe, model)
  tibble(
    accuracy = accuracy,
    units1 = units1,
    units2 = units2,
    act1 = act1,
    act2 = act2,
    act3 = act3
  )
}

#' @title Retrain the best model.
#' @description After we find the model with the best accuracy,
#'   retrain it and return the trained model given a row of output
#'   from [test_model()].
#' @details The first time you run Keras in an R session,
#'   TensorFlow usually prints verbose ouput such as
#'   "Your CPU supports instructions that this TensorFlow binary was not compiled to use:"
#'   and "OMP: Info #171: KMP_AFFINITY:". You can safely ignore these messages.
#' @return A trained Keras model.
#' @inheritParams best_run A one-row data frame from [test_model()]
#'   corresponding to the best model.
#' @inheritParams test_accuracy
#' @examples
#' library(keras)
#' library(recipes)
#' library(rsample)
#' library(tidyverse)
#' data <- split_data("data/customer_churn.csv")
#' recipe <- prepare_recipe(data)
#' run <- test_model(data, recipe, 16, 16, "relu", "relu", "sigmoid")
#' train_best_model(run, recipe)
train_best_model <- function(best_run, churn_recipe) {
  train_model(
    churn_recipe,
    best_run$units1,
    best_run$units2,
    best_run$act1,
    best_run$act2,
    best_run$act3
  )
}
