library(stringr)
library(dplyr)
library(tfdatasets)
library(keras)
library(alluvial)

files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/",
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


#ds <- tensor_slices_dataset(df)

window_size_ms <- 30
window_stride_ms <- 10

# window_size <- as.integer(16000*window_size_ms/1000)
# stride <- as.integer(16000*window_stride_ms/1000)
#
# fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
# n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))
#
# # shortcuts to used TensorFlow modules.
# audio_ops <- tf$contrib$framework$python$ops$audio_ops
#
# ds <- ds %>%
#   dataset_map(function(obs) {
#
#     # a good way to debug when building tfdatsets pipelines is to use a print
#     # statement like this:
#     # print(str(obs))
#
#     # decoding wav files
#     audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
#     wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
#
#     # create the spectrogram
#     spectrogram <- audio_ops$audio_spectrogram(
#       wav$audio,
#       window_size = window_size,
#       stride = stride,
#       magnitude_squared = TRUE
#     )
#
#     # normalization
#     spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
#
#     # moving channels to last dim
#     spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
#
#     # transform the class_id into a one-hot encoded vector
#     response <- tf$one_hot(obs$class_id, 30L)
#
#     list(spectrogram, response)
#   })

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

set.seed(6)
id_train <- sample(nrow(df), size = 0.7*nrow(df))

ds_train <- data_generator(
  df[id_train,],
  batch_size = 32,
  window_size_ms = 30,
  window_stride_ms = 10
)

ds_validation <- data_generator(
  df[-id_train,],
  batch_size = 32,
  shuffle = FALSE,
  window_size_ms = 30,
  window_stride_ms = 10
)


window_size <- as.integer(16000*window_size_ms/1000)
stride <- as.integer(16000*window_stride_ms/1000)
fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))


model <- keras_model_sequential()
model %>%
  layer_conv_2d(input_shape = c(n_chunks, fft_size, 1),
                filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 30, activation = 'softmax')



model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)



model %>% fit(
  generator = ds_train,
  steps_per_epoch = 0.7*nrow(df)/32,
  epochs = 10,
  validation_data = ds_validation,
  validation_steps = 0.3*nrow(df)/32
)


df_validation <- df[-id_train,]
n_steps <- nrow(df_validation)/32 + 1



predictions <- predict(
  model,
  ds_validation,
  steps = n_steps
)

str(predictions)



x <- df_validation %>%
  mutate(pred_class_id = head(classes, nrow(df_validation))) %>%
  left_join(
    df_validation %>% distinct(class_id, class) %>% rename(pred_class = class),
    by = c("pred_class_id" = "class_id")
  ) %>%
  mutate(correct = pred_class == class) %>%
  count(pred_class, class, correct)

#
# alluvial(
#   x %>% select(class, pred_class),
#   freq = x$n,
#   col = ifelse(x$correct, "lightblue", "red"),
#   border = ifelse(x$correct, "lightblue", "red"),
#   alpha = 0.6,
#   hide = x$n < 20
# )
