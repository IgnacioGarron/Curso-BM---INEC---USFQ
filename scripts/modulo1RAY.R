# ===============================================================
# 1. Configuración inicial del entorno de trabajo
# ===============================================================
# Se define un vector con los paquetes requeridos para el análisis.
# Nota: aquí solo se listan; la carga efectiva se realiza en el bloque siguiente.
paquetes <- c("tidymodels", "tidyverse", "broom", "glmnet", "readxl")


# Se cargan las librerías necesarias sin mostrar mensajes de inicio.
# Esto mantiene la salida de consola más limpia y facilita leer los resultados importantes.
suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(broom)
  library(readxl)
})


# Se da prioridad a las funciones de tidymodels cuando existan nombres repetidos
# con otros paquetes cargados. Esto ayuda a evitar conflictos entre funciones.
tidymodels_prefer()

# Se establece un tema gráfico global minimalista para los gráficos de ggplot2.
# El tamaño base de letra queda definido en 12 para mejorar legibilidad.
theme_set(theme_minimal(base_size = 12))


# Se fija una semilla global para garantizar reproducibilidad.
# Con la misma semilla, las particiones aleatorias y remuestreos serán replicables.
semilla_global <- 123
set.seed(semilla_global)


# Se define la carpeta donde se guardarán los resultados generados por el script.
output_dir <- file.path("output", "modulo1RAY")
# Se crea la carpeta de salida si no existe.
# recursive = TRUE permite crear subcarpetas intermedias.
# showWarnings = FALSE evita advertencias si la carpeta ya existe.
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Cargar datos reales del INEC y preparar nombres sencillos para modelar.

# ===============================================================
# 2. Carga y preparación inicial de los datos
# ===============================================================
# Se lee el archivo Excel con la base original de pobreza del INEC.
# suppressMessages() evita mensajes adicionales durante la lectura del archivo.
data_inec_original <- suppressMessages(read_excel("data/data_pobreza_INEC.xlsx"))
# Se crea una copia de trabajo para conservar intacta la base original cargada.
data_inec <- data_inec_original
# Se transforman los nombres de columnas a nombres válidos en R.
# unique = TRUE asegura que no existan nombres duplicados después de la transformación.
names(data_inec) <- make.names(names(data_inec), unique = TRUE)


# Se construye la base analítica que será usada en el resto del ejercicio.
# Aquí se renombran variables clave, se crea la variable objetivo categórica
# y se elimina una columna que no se utilizará en el análisis posterior.
datos_pobreza <- data_inec %>%
  rename(
    canton = Canton,
    id_canton = CodCanton...2,
    nbi_mef = NBIMEF
  ) %>%
  mutate(
    canton = as.character(canton),
    pobre = factor(
      if_else(nbi_mef >= median(nbi_mef, na.rm = TRUE), "Pobre", "No pobre"),
      levels = c("No pobre", "Pobre")
    )
  ) %>%
  select(-CodCanton...3)


# Se inspecciona la estructura de la base resultante:
# número de filas, columnas, tipos de variables y algunos valores de ejemplo.
datos_pobreza %>% glimpse()


# ===============================================================
# 3. Resumen descriptivo y diagnóstico de datos faltantes
# ===============================================================
# Se calcula un resumen general de la base:
# número de cantones, promedio y mediana del NBI MEF, y tasa de pobreza construida.
resumen_pobreza <- datos_pobreza %>%
  summarise(
    n = n(),
    nbi_promedio = mean(nbi_mef, na.rm = TRUE),
    nbi_mediano = median(nbi_mef, na.rm = TRUE),
    tasa_pobreza = mean(pobre == "Pobre", na.rm = TRUE)
  )


# Se identifica cuántos valores faltantes tiene cada variable.
# Luego se reorganiza la tabla en formato largo y se conservan solo variables con faltantes.
missing_pobreza <- datos_pobreza %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  filter(n_missing > 0)


# Se exporta el resumen descriptivo a un archivo CSV para documentación y trazabilidad.
write_csv(resumen_pobreza, file.path(output_dir, "resumen_pobreza.csv"))
# Se exporta el diagnóstico de datos faltantes a un archivo CSV.
write_csv(missing_pobreza, file.path(output_dir, "missing_pobreza.csv"))


# Se imprime en consola el resumen descriptivo calculado.
resumen_pobreza
# Se imprime en consola el listado de variables con datos faltantes.
missing_pobreza


# ===============================================================
# 4. Imputación de valores faltantes
# ===============================================================
# Función auxiliar para imputar categorías con la moda.
# Esta función calcula la moda de un vector, es decir, el valor más frecuente.
# Se usará para imputar variables categóricas cuando tengan valores faltantes.
moda <- function(x) {
  valores <- x[!is.na(x)]
  if (length(valores) == 0) return(NA)
  valores[which.max(tabulate(match(valores, unique(valores))))]
}


# Se crea una versión imputada de la base:
# las variables numéricas se completan con su media,
# y las variables de texto o factores se completan con su moda.
datos_pobreza_imputados <- datos_pobreza %>%
  mutate(
    across(where(is.numeric), ~ replace_na(.x, mean(.x, na.rm = TRUE))),
    across(where(~ is.character(.x) || is.factor(.x)), ~ replace_na(.x, moda(.x)))
  )


# Se guarda la base imputada en CSV para conservar una versión limpia del insumo analítico.
write_csv(datos_pobreza_imputados, file.path(output_dir, "datos_pobreza_imputados_media_moda.csv"))


# Se verifica si todavía quedan valores faltantes después de la imputación.
datos_pobreza_imputados %>% summarise(across(everything(), ~ sum(is.na(.x))))


# ===============================================================
# 5. Normalización exploratoria de variables numéricas
# ===============================================================
# Se seleccionan las variables numéricas que podrían servir como predictores.
# Se excluyen identificadores y variables objetivo o de referencia que no deben normalizarse como predictores.
variables_numericas_modelo <- datos_pobreza_imputados %>%
  select(where(is.numeric), -id_canton, -nbi_mef, -NBIenemdu, -nbicenso2022) %>%
  names()


# Se calculan los parámetros de normalización de cada variable numérica seleccionada:
# media y desviación estándar.
parametros_normalizacion <- datos_pobreza_imputados %>%
  summarise(across(
    all_of(variables_numericas_modelo),
    list(media = mean, desviacion = sd),
    .names = "{.col}_{.fn}"
  )) %>%
  pivot_longer(
    everything(),
    names_to = c("variable", "estadistico"),
    names_pattern = "(.+)_(media|desviacion)",
    values_to = "valor"
  ) %>%
  pivot_wider(names_from = estadistico, values_from = valor)


# Se agregan columnas normalizadas tipo z-score para cada variable numérica seleccionada.
# Cada nueva columna representa cuántas desviaciones estándar está el valor respecto a su media.
datos_pobreza_normalizados <- datos_pobreza_imputados %>%
  mutate(across(
    all_of(variables_numericas_modelo),
    ~ (.x - mean(.x)) / sd(.x),
    .names = "{.col}_z"
  ))


# Se guardan los parámetros de normalización para poder reproducir la transformación.
write_csv(parametros_normalizacion, file.path(output_dir, "parametros_normalizacion.csv"))
# Se guarda la base con variables normalizadas.
write_csv(datos_pobreza_normalizados, file.path(output_dir, "datos_pobreza_normalizados.csv"))


# Se imprime la tabla de medias y desviaciones estándar usadas para normalizar.
parametros_normalizacion

# Se revisa la estructura de la base normalizada, mostrando identificadores,
# variable objetivo y columnas terminadas en _z.
datos_pobreza_normalizados %>%
  select(canton, id_canton, nbi_mef, pobre, ends_with("_z")) %>%
  glimpse()


# ===============================================================
# 6. Visualización descriptiva de la distribución de pobreza
# ===============================================================
# Se construye un histograma del NBI MEF coloreado por la condición de pobreza creada.
datos_pobreza %>%
  ggplot(aes(x = nbi_mef, fill = pobre)) +
  geom_histogram(bins = round(nrow(datos_pobreza)^(1 / 2), 0), alpha = .75, color = "white") +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "NBI MEF", y = "Cantones", fill = "Condición", title = "Distribución de pobreza por NBI")


# ===============================================================
# 7. Particiones, validación cruzada y bootstrap
# ===============================================================
# Se calcula la proporción real de cantones clasificados como pobres en la base completa.
# Esta proporción servirá como referencia para comparar distintos esquemas de partición.
prop_pobre_real <- mean(datos_pobreza$pobre == "Pobre")


# Se vuelve a fijar la semilla antes de la partición simple para que el resultado sea reproducible.
set.seed(semilla_global)
# Se divide la base en entrenamiento y prueba usando 80% para entrenamiento.
# Esta partición no controla explícitamente la proporción de la clase pobre.
split_simple <- initial_split(datos_pobreza, prop = 0.80)
# Se extrae el conjunto de entrenamiento de la partición simple.
train_simple <- training(split_simple)
# Se extrae el conjunto de prueba de la partición simple.
test_simple <- testing(split_simple)


# Se vuelve a fijar la semilla antes de la partición estratificada.
# La estratificación busca preservar la proporción de pobres y no pobres en train y test.
set.seed(semilla_global)
# Se crea una partición estratificada por la variable pobre.
# Esto ayuda a que train y test mantengan una proporción similar de pobreza a la base completa.
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

# Se obtiene la base de entrenamiento de la partición estratificada.
train <- training(split_pobreza)
# Se obtiene la base de prueba de la partición estratificada.
test <- testing(split_pobreza)


# Se fija la semilla para generar validación cruzada simple reproducible.
set.seed(semilla_global)
# Se generan 5 folds de validación cruzada sin estratificación.
folds_simple <- vfold_cv(datos_pobreza, v = 5)


# Se fija la semilla para generar validación cruzada estratificada reproducible.
set.seed(semilla_global)
# Se generan 5 folds de validación cruzada estratificada por condición de pobreza.
folds_strata <- vfold_cv(datos_pobreza, v = 5, strata = pobre)


# Se fija la semilla para generar muestras bootstrap simples reproducibles.
set.seed(semilla_global)
# Se generan 5 muestras bootstrap simples.
boots_simple <- bootstraps(datos_pobreza, times = 5)


# Se fija la semilla para generar muestras bootstrap estratificadas reproducibles.
set.seed(semilla_global)
# Se generan 5 muestras bootstrap estratificadas por condición de pobreza.
boots_strata <- bootstraps(datos_pobreza, times = 5, strata = pobre)


# Se arma una tabla comparativa de la proporción de pobreza en cada esquema de partición.
# Esto permite evaluar cuánto se aleja cada muestra de la proporción observada en la base completa.
resumen_particion <- tibble(
  particion = c(
    "Datos completos",
    "Initial split simple - Train",
    "Initial split simple - Test",
    "Initial split con strata - Train",
    "Initial split con strata - Test",
    "K-fold simple - Fold 1",
    "K-fold con strata - Fold 1",
    "Bootstrap simple - Muestra 1",
    "Bootstrap con strata - Muestra 1"
  ),
  prop_pobre = c(
    prop_pobre_real,
    mean(train_simple$pobre == "Pobre"),
    mean(test_simple$pobre == "Pobre"),
    mean(train$pobre == "Pobre"),
    mean(test$pobre == "Pobre"),
    mean(assessment(folds_simple$splits[[1]])$pobre == "Pobre"),
    mean(assessment(folds_strata$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_simple$splits[[1]])$pobre == "Pobre"),
    mean(analysis(boots_strata$splits[[1]])$pobre == "Pobre")
  )
) %>%
  mutate(
    prop_real = prop_pobre_real,
    diferencia_vs_real = prop_pobre - prop_real
  )


# Se guarda la comparación de particiones en un archivo CSV.
write_csv(resumen_particion, file.path(output_dir, "resumen_particion.csv"))


# Se imprime la tabla de comparación de proporciones por partición.
resumen_particion


# ===============================================================
# 8. Selección de predictores para los modelos
# ===============================================================
# Se define manualmente el conjunto de variables explicativas que usarán los modelos.
# Estas variables representan dimensiones socioeconómicas, financieras, educativas, demográficas y de servicios.
predictores_modelo <- c(
  "PorcentajeInstConInternet",
  "TEF11a19madre",
  "CajerosAutomáticosTasapobmay15años",
  "TotPuntosAteFinTasapobmay15años",
  "PIBpercap",
  "TasaGlobalFecundmadre",
  "ParticipacionPIBEnseñanza",
  "Porcentajenacprematuromoderadomujermujermadre",
  "aguaPotableViv",
  "OficinasTasapobmay15años"
)
# Se transforman los nombres de predictores al mismo formato válido en R usado en la base.
# Esto evita errores por espacios, tildes, símbolos o nombres duplicados.
predictores_modelo <- make.names(predictores_modelo, unique = TRUE)


# Se reinicia la semilla antes de volver a generar la partición train/test estratificada.
# Esto asegura que el bloque de modelamiento use una partición reproducible.
set.seed(semilla_global)
# Se repite la partición estratificada para el bloque de modelamiento que sigue.
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

# Se obtiene la base de entrenamiento de la partición estratificada.
train <- training(split_pobreza)
# Se obtiene la base de prueba de la partición estratificada.
test <- testing(split_pobreza)


# ===============================================================
# 9. Modelo de regresión lineal para predecir NBI MEF
# ===============================================================
# Se define una receta de preprocesamiento para el modelo lineal.
# La variable respuesta continua es nbi_mef y los predictores son los definidos arriba.
receta_ols <- recipe(
  reformulate(predictores_modelo, response = "nbi_mef"),
  data = train
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Se especifica un modelo de regresión lineal ordinaria.
# El motor lm corresponde a la función base de R para modelos lineales.
modelo_ols <- linear_reg() %>%
  set_engine("lm")


# Se construye un workflow que une la receta de preprocesamiento con el modelo lineal.
# En tidymodels, el workflow centraliza el entrenamiento y la predicción.
wf_ols <- workflow() %>%
  add_recipe(receta_ols) %>%
  add_model(modelo_ols)


# Se imprime el workflow para revisar su estructura antes del ajuste.
wf_ols


# Se entrena el modelo lineal usando la base de entrenamiento.
ajuste_ols <- fit(wf_ols, data = train)


# Se generan predicciones de NBI MEF sobre la base de prueba.
# Luego se unen las predicciones con los valores reales para evaluar desempeño.
pred_ols <- predict(ajuste_ols, new_data = test) %>%
  bind_cols(test %>% select(nbi_mef))


# Se muestran las primeras predicciones del modelo lineal.
head(pred_ols)


# Se define el conjunto de métricas para evaluar el modelo de regresión:
# RMSE mide error cuadrático promedio, R² mide ajuste explicado y MAE mide error absoluto promedio.
metr <- metric_set(rmse, rsq, mae)


# Se calculan las métricas del modelo lineal comparando valores reales y predichos.
metricas_ols <- metr(
  pred_ols,
  truth = nbi_mef,
  estimate = .pred
)


# Se guardan las métricas del modelo lineal en un archivo CSV.
write_csv(metricas_ols, file.path(output_dir, "metricas_ols.csv"))


# Se reinicia la semilla antes de volver a generar la partición train/test estratificada.
# Esto asegura que el bloque de modelamiento use una partición reproducible.
set.seed(semilla_global)
# Se repite la partición estratificada para el bloque de modelamiento que sigue.
split_pobreza <- initial_split(datos_pobreza, prop = 0.80, strata = pobre)

# Se obtiene la base de entrenamiento de la partición estratificada.
train <- training(split_pobreza)
# Se obtiene la base de prueba de la partición estratificada.
test <- testing(split_pobreza)


# ===============================================================
# 10. Modelo logístico para clasificar cantones pobres/no pobres
# ===============================================================
# Se define la receta para el modelo de clasificación.
# La variable respuesta categórica es pobre.
receta_logit <- recipe(
  reformulate(predictores_modelo, response = "pobre"),
  data = train
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Se especifica un modelo de regresión logística.
# El motor glm ajusta el modelo usando la implementación clásica de R.
modelo_logit <- logistic_reg() %>%
  set_engine("glm")


# Se construye el workflow que combina la receta logística con el modelo logístico.
wf_logit <- workflow() %>%
  add_recipe(receta_logit) %>%
  add_model(modelo_logit)


# Se imprime el workflow logístico para revisar su configuración.
wf_logit


# Se entrena el modelo logístico con la base de entrenamiento.
ajuste_logit <- fit(wf_logit, data = train)


# Se predice sobre la base de prueba:
# primero probabilidades por clase, luego clase predicha, y finalmente se une la clase real.
pred_logit <- predict(ajuste_logit, new_data = test, type = "prob") %>%
  bind_cols(predict(ajuste_logit, new_data = test, type = "class")) %>%
  bind_cols(test %>% select(pobre))


# Se muestran las primeras predicciones del modelo logístico.
head(pred_logit)


# Se define el conjunto de métricas para clasificación:
# exactitud, precisión, recall, F1 y AUC ROC.
metr <- metric_set(
  accuracy,
  precision,
  recall,
  f_meas,
  roc_auc
)


# Se calculan las métricas del modelo logístico.
# event_level = 'second' indica que la clase positiva será 'Pobre',
# porque los niveles del factor fueron definidos como 'No pobre' y luego 'Pobre'.
metricas_logit <- metr(
  pred_logit,
  truth = pobre,
  estimate = .pred_class,
  .pred_Pobre,
  event_level = "second"
)


# Se calcula la matriz de confusión para comparar clases reales y predichas en test.
matriz_confusion <- conf_mat(
  pred_logit,
  truth = pobre,
  estimate = .pred_class
)


# Se guardan las métricas del modelo logístico.
write_csv(metricas_logit, file.path(output_dir, "metricas_logit.csv"))
# Se guarda la matriz de confusión del conjunto de prueba en formato tabular.
write_csv(tidy(matriz_confusion), file.path(output_dir, "matriz_confusion_logit.csv"))


# Se imprimen las métricas del modelo logístico.
metricas_logit
# Se imprime la matriz de confusión del modelo logístico.
matriz_confusion


# ===============================================================
# 11. Evaluación adicional en entrenamiento y prueba
# ===============================================================
# Ejercicio final: predecir los cantones con mayor pobreza por NBI.
# NBIMEF es la variable objetivo continua; "pobre" identifica cantones con NBI
# igual o superior a la mediana observada.


# Se generan predicciones del modelo logístico sobre la base de entrenamiento.
# Esto permite comparar el desempeño en entrenamiento frente al desempeño en prueba.
pred_logit_train <- predict(ajuste_logit, new_data = train, type = "prob") %>%
  bind_cols(predict(ajuste_logit, new_data = train, type = "class")) %>%
  bind_cols(train %>% select(pobre))


# Se calcula la matriz de confusión para el conjunto de entrenamiento.
matriz_confusion_train <- conf_mat(
  pred_logit_train,
  truth = pobre,
  estimate = .pred_class
)


# Se reutiliza la matriz de confusión ya calculada para el conjunto de prueba.
matriz_confusion_test <- matriz_confusion


# Se guarda la matriz de confusión del entrenamiento.
write_csv(tidy(matriz_confusion_train), file.path(output_dir, "matriz_confusion_logit_train.csv"))
# Se guarda la matriz de confusión de prueba.
write_csv(tidy(matriz_confusion_test), file.path(output_dir, "matriz_confusion_logit_test.csv"))


# Se imprime la matriz de confusión del entrenamiento.
matriz_confusion_train
# Se imprime la matriz de confusión de prueba.
matriz_confusion_test


# ===============================================================
# 12. Modelos finales y ranking operativo de pobreza cantonal
# ===============================================================
# Modelos finales entrenados con todos los cantones disponibles para producir
# el ranking operativo de pobreza cantonal.
# Se define la receta final del modelo lineal usando todos los datos disponibles.
# Esta versión ya no separa train/test porque se usa para producir predicciones operativas.
receta_ols_final <- recipe(
  reformulate(predictores_modelo, response = "nbi_mef"),
  data = datos_pobreza
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Se construye el workflow final del modelo lineal.
wf_ols_final <- workflow() %>%
  add_recipe(receta_ols_final) %>%
  add_model(modelo_ols)


# Se entrena el modelo lineal final con todos los cantones disponibles.
ajuste_ols_final <- fit(wf_ols_final, data = datos_pobreza)


# Se define la receta final del modelo logístico usando todos los datos disponibles.
receta_logit_final <- recipe(
  reformulate(predictores_modelo, response = "pobre"),
  data = datos_pobreza
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Se construye el workflow final del modelo logístico.
wf_logit_final <- workflow() %>%
  add_recipe(receta_logit_final) %>%
  add_model(modelo_logit)


# Se entrena el modelo logístico final con todos los cantones disponibles.
ajuste_logit_final <- fit(wf_logit_final, data = datos_pobreza)


# Se genera una tabla final de predicciones por cantón.
# Incluye el NBI observado, la pobreza observada, el NBI predicho, probabilidades de pobreza,
# clase predicha, error de predicción y ranking de pobreza predicha.
predicciones_cantones <- datos_pobreza %>%
  select(canton, id_canton, nbi_mef, pobre) %>%
  bind_cols(predict(ajuste_ols_final, new_data = datos_pobreza) %>% rename(nbi_mef_predicha = .pred)) %>%
  bind_cols(predict(ajuste_logit_final, new_data = datos_pobreza, type = "prob")) %>%
  bind_cols(predict(ajuste_logit_final, new_data = datos_pobreza, type = "class") %>% rename(pobre_predicho = .pred_class)) %>%
  mutate(
    error_nbi = nbi_mef - nbi_mef_predicha,
    ranking_pobreza_predicha = min_rank(desc(nbi_mef_predicha))
  ) %>%
  arrange(ranking_pobreza_predicha)


# Se extraen los 20 cantones con mayor pobreza predicha según el ranking del modelo lineal.
cantones_mas_pobres_predichos <- predicciones_cantones %>%
  slice_head(n = 20)


# Se calcula una matriz de confusión usando las predicciones finales sobre todos los cantones.
# Esta matriz resume la concordancia entre la clasificación observada y la clasificación predicha.
matriz_confusion_todos_cantones <- conf_mat(
  predicciones_cantones,
  truth = pobre,
  estimate = pobre_predicho
)


# Se guarda la tabla completa de predicciones cantonales.
write_csv(predicciones_cantones, file.path(output_dir, "predicciones_todos_cantones.csv"))
# Se guarda el listado de los 20 cantones con mayor pobreza predicha.
write_csv(cantones_mas_pobres_predichos, file.path(output_dir, "cantones_mas_pobres_predichos.csv"))
# Se guarda la matriz de confusión final para todos los cantones.
write_csv(tidy(matriz_confusion_todos_cantones), file.path(output_dir, "matriz_confusion_logit_todos_cantones.csv"))


# Se imprime el ranking de los 20 cantones con mayor pobreza predicha.
cantones_mas_pobres_predichos
# Se imprime la matriz de confusión calculada con todos los cantones.
matriz_confusion_todos_cantones
