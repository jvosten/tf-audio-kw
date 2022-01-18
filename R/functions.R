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

#' @title Train set sampler
#' @description generates a sample of IDs of 70% of the data set for
#' the data set split
#' @return A vector
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @examples
#' tbc
id_sampler <- function(df) {
  as_tibble(as.numeric(sample(nrow(df), size = 0.7*nrow(df))))
}

#' @title Dataset generator
#' @description define a function called data_generator that will create the
#' generator depending on specified inputs.
#' @return A function that defines how the TensorFlow graph should read
#' and pre-process data. Not compiled or run yet.
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @param batch_size
#' @param window_size_ms
#' @param window_stride_ms
#' @param shuffle
#' @examples
#' library(tfdatasets)
data_generator <- function(df,
                           batch_size = 32,
                           window_size_ms,
                           window_stride_ms,
                           shuffle = TRUE) {

  # assume sampling rate is the same in all samples
  sampling_rate <-tensorflow::tf$cast(16000, tensorflow::tf$int32)

  samples_per_window <- (sampling_rate * window_size_ms) %/% 1000L
  stride_samples <-  (sampling_rate * window_stride_ms) %/% 1000L

  n_periods <-
    tensorflow::tf$shape(
      tensorflow::tf$range(
        samples_per_window %/% 2L,
        16000L - samples_per_window %/% 2L,
        stride_samples
      )
    )[1] + 1L

  n_fft_coefs <- tensorflow::tf$cast(
    (2 ^ tensorflow::tf$math$ceil(tensorflow::tf$math$log(
      tensorflow::tf$cast(samples_per_window, tensorflow::tf$float32)
    ) / tensorflow::tf$math$log(2)) /
      2 + 1L),
    tensorflow::tf$int32)

  ds <- tfdatasets::tensor_slices_dataset(df)

  if (shuffle == TRUE){
    ds <- ds %>%
      dataset_shuffle(buffer_size = 100)
  }

  ds <- ds %>%
    dataset_map(function(obs) {
      wav <-
        tensorflow::tf$audio$decode_wav(tensorflow::tf$io$read_file(tensorflow::tf$reshape(obs$fname, list())))
      samples <- wav$audio
      samples <- samples %>% tensorflow::tf$transpose(perm = c(1L, 0L))

      stft_out <- tensorflow::tf$signal$stft(samples,
                                             frame_length = samples_per_window,
                                             frame_step = stride_samples)

      magnitude_spectrograms <- tensorflow::tf$abs(stft_out)
      log_magnitude_spectrograms <- tensorflow::tf$math$log(magnitude_spectrograms + 1e-6)

      response <- tensorflow::tf$one_hot(obs$class_id, 30L)

      input <- tensorflow::tf$transpose(log_magnitude_spectrograms, perm = c(1L, 2L, 0L))
      list(input, response)
    })

  ds <- ds %>%
    dataset_repeat()

  ds <- ds %>%
    dataset_padded_batch(
      batch_size = batch_size,
      padded_shapes = list(
        tensorflow::tf$stack(list(
          n_periods,
          n_fft_coefs,-1L)),
        tensorflow::tf$constant(-1L,
                                shape = shape(1L))),
      drop_remainder = TRUE
    )

  ds
}

#' @title Test set generator
#' @description define a function called ds_train that will create the
#' generator depending on specified inputs.
#' @return A tf_dataset object defines how the TensorFlow graph should read
#' and pre-process test data. Not compiled or run yet.
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @param batch_size
#' @param window_size_ms
#' @param window_stride_ms
#' @param shuffle
#' @examples
#' library(tfdatasets)
ds_train <- function(
  df,
  id_train,
  batch_size = 32,
  window_size_ms,
  window_stride_ms,
  shuffle
) {
  df <- df[pull(id_train),]

  data_generator(
    df,
    batch_size = 32,
    window_size_ms,
    window_stride_ms,
    shuffle = TRUE
  )
}

#' @title Validation set generator
#' @description Define a function called ds_validation that will create the
#' generator depending on specified inputs.
#' @return A tf_dataset object defines how the TensorFlow graph should read
#' and pre-process validation data. Not compiled or run yet.
#' @param generator_prep A `tibble` generated from [prep_tbl()]
#' @param batch_size
#' @param window_size_ms
#' @param window_stride_ms
#' @param shuffle
#' @examples
#' library(tfdatasets)
#' tbc
ds_validation <- function(
  df,
  id_train,
  batch_size = 32,
  window_size_ms,
  window_stride_ms,
  shuffle = FALSE
) {
  df <- df[-pull(id_train),]

  data_generator(
    df,
    batch_size = 32,
    window_size_ms,
    window_stride_ms,
    shuffle = FALSE
  )
}

#' @title Define a Keras model.
#' @description Define a Keras model for the customer churn data.
#' @return A Keras model. Not compiled or run yet.
#' @param window_size_ms
#' @param window_stride_ms
#' @examples
#' library(keras)
#' library(tfdatasets)
#' model <- define_model(n_chunks, fft_size)
#' model
define_model <- function(
  window_size_ms,
  window_stride_ms
) {
  window_size <- as.integer(16000*window_size_ms/1000)
  stride <- as.integer(16000*window_stride_ms/1000)
  fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
  n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))

  keras_model_sequential() %>%
    layer_conv_2d(
      input_shape = c(n_chunks, fft_size, 1),
      filters = 32,
      kernel_size = c(3,3),
      activation = 'relu'
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 64,
      kernel_size = c(3,3),
      activation = 'relu'
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 128,
      kernel_size = c(3,3),
      activation = 'relu'
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 256,
      kernel_size = c(3,3),
      activation = 'relu'
    ) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(
      units = 128,
      activation = 'relu'
    ) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(
      units = 30,
      activation = 'softmax'
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
#' @param df
#' @param ds_train
#' @param ds_validate
#' @examples
#' library(keras)
#' library(tfdatasets)
#' tbc
train_model <- function(
  df,
  id_train,
  batch_size,
  window_size_ms,
  window_stride_ms
) {
  model <- define_model(window_size_ms,
                        window_stride_ms)

  ds_train <- ds_train(
    df = df,
    id_train = id_train,
    batch_size = batch_size,
    window_size_ms = window_size_ms,
    window_stride_ms = window_stride_ms,
    shuffle = TRUE
  )

  ds_validation <- ds_validation(
    df = df,
    id_train = id_train,
    batch_size = batch_size,
    window_size_ms = window_size_ms,
    window_stride_ms = window_stride_ms,
    shuffle = FALSE
  )

  compile(
    model,
    loss = loss_categorical_crossentropy,
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy')
  )

  model %>% fit_generator(
    generator = ds_train,
    steps_per_epoch = 0.7*nrow(df)/32,
    epochs = 10,
    validation_data = ds_validation,
    validation_steps = 0.3*nrow(df)/32
  )
  model
}

#' @title Apply the trained model on validation dataset
#' @description ...
#' @details ...
#' @return Confusion matrix
#' @inheritParams train_model
#' @inheritParams
#' @examples
predicting <- function(
  df,
  id_train,
  batch_size,
  window_size_ms,
  window_stride_ms,
  model
) {
  ds_validation <- ds_validation(
    df = df,
    id_train = id_train,
    batch_size = batch_size,
    window_size_ms = window_size_ms,
    window_stride_ms = window_stride_ms,
    shuffle = FALSE
  )

  df_validation <- df[-pull(id_train),]
  n_steps <- nrow(df_validation)/32 + 1

  predictions <- keras::predict_generator(
    model,
    ds_validation,
    steps = n_steps
  )

  classes <- apply(predictions, 1, which.max) - 1

  confusion_mat <- df_validation %>%
    mutate(pred_class_id = head(classes, nrow(df_validation))) %>%
    left_join(
      df_validation %>% distinct(class_id, class) %>% rename(pred_class = class),
      by = c("pred_class_id" = "class_id")
    ) %>%
    mutate(correct = pred_class == class)
}

#' @title Visualization
#' @description Creates an alluvial plot
#' @details A nice visualization of the confusion matrix is to create an alluvial diagram:
#' @return Plot
#' @inheritParams predicting
#' @inheritParams test_accuracy
#' @examples
#' library(alluvial)
#' tbc
alluv_plot <- function(conf_mat, path) {
  x <- conf_mat %>%
    count(pred_class, class, correct)

  save_plot <- function(x, path) {
    png('output/alluvial.png') #width = 1920,height = 1080
    invisible(x)
    invisible(dev.off())
  }

  alluvial(
    x %>% select(class, pred_class),
    freq = x$n,
    col = ifelse(x$correct, "lightblue", "red"),
    border = ifelse(x$correct, "lightblue", "red"),
    alpha = 0.6,
    hide = x$n < 20
  ) %>%
    save_plot(x = ., path)
}
