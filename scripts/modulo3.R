# ============================================================
# MÓDULO 3
# Comparación de varios modelos de clasificación:
# SVM radial, k-NN, Naive Bayes, Random Forest y GBM/XGBoost.
#
# Objetivo general:
# - Usar datos de hogares para clasificar pobreza.
# - Comparar modelos mediante validación cruzada.
# - Seleccionar un modelo candidato usando PR AUC.
# - Evaluar el modelo candidato en test.
# - Construir curvas de validación y curvas de aprendizaje.
# - Dejar planteado un ejercicio adicional con datos cantonales INEC.
# ============================================================


# ------------------------------------------------------------
# Se define un vector con todos los paquetes necesarios.
#
# Esta lista funciona como inventario de dependencias del módulo.
# Incluye paquetes del ecosistema tidymodels, modelos específicos
# y herramientas de interpretación/visualización.
# ------------------------------------------------------------
paquetes_necesarios <- c(
  "tidymodels", "tidyverse", "recipes", "parsnip", "tune", "yardstick",
  "readxl", "vip", "DALEXtra", "kernlab", "kknn",
  "naivebayes", "discrim", "ranger", "xgboost", "factoextra"
)


# ------------------------------------------------------------
# Se identifica qué paquetes no están instalados.
#
# requireNamespace(..., quietly = TRUE) revisa si cada paquete
# está disponible sin cargarlo completamente.
#
# vapply() aplica esa revisión a todos los paquetes y devuelve
# un vector lógico.
# ------------------------------------------------------------
paquetes_faltantes <- paquetes_necesarios[!vapply(paquetes_necesarios, requireNamespace, logical(1), quietly = TRUE)]
paquetes_faltantes


# ------------------------------------------------------------
# Si aparece algún paquete faltante, se puede instalar.
#
# La línea está comentada para evitar instalaciones automáticas.
# El estudiante o analista debe descomentar si realmente necesita
# instalar dependencias.
# ------------------------------------------------------------
# Si aparece algún paquete faltante, descomentar y ejecutar:
# install.packages(paquetes_faltantes)


# ------------------------------------------------------------
# Se cargan los paquetes principales.
#
# suppressPackageStartupMessages() evita que aparezcan mensajes largos
# de carga en la consola.
#
# tidymodels: modelado predictivo.
# tidyverse: manipulación y visualización de datos.
# readxl: lectura de Excel.
# discrim: modelos discriminantes y Naive Bayes dentro de tidymodels.
# vip: importancia de variables.
# DALEXtra: explicabilidad de modelos.
# factoextra: visualización de métodos factoriales y clustering.
# ------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(readxl)
  library(discrim)
  library(vip)
  library(DALEXtra)
  library(factoextra)
})


# ------------------------------------------------------------
# tidymodels_prefer() resuelve conflictos de funciones.
# Cuando varios paquetes tienen funciones con el mismo nombre,
# se priorizan las versiones compatibles con tidymodels.
# ------------------------------------------------------------
tidymodels_prefer()


# ------------------------------------------------------------
# Se fija un tema visual minimalista para ggplot2.
# base_size = 12 define el tamaño base de las fuentes.
# ------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))


# ------------------------------------------------------------
# Se define una semilla global para reproducibilidad.
# Esto permite repetir particiones, validaciones y modelos
# con resultados consistentes.
# ------------------------------------------------------------
semilla_global <- 123
set.seed(semilla_global)


# ------------------------------------------------------------
# Se define la carpeta de salida del módulo 3.
# Aquí se guardarán métricas, predicciones y tablas generadas.
#
# recursive = TRUE permite crear carpetas anidadas.
# showWarnings = FALSE evita advertencias si la carpeta ya existe.
# ------------------------------------------------------------
output_dir <- file.path("output", "modulo3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# Se cargan los datos de hogares preparados previamente.
#
# show_col_types = FALSE evita imprimir el diagnóstico de tipos
# de columnas producido por read_csv().
#
# Luego se convierten variables categóricas a factor.
# Finalmente, pobre se declara como factor con niveles explícitos:
# primero "No pobre" y luego "Pobre".
# ------------------------------------------------------------
hogares <- read_csv("data/hogares_modulo1.csv", show_col_types = FALSE) %>%
  mutate(
    across(c(region, area, educ_jefe, empleo_jefe, vivienda_propia, internet), as.factor),
    pobre = factor(pobre, levels = c("No pobre", "Pobre"))
  )


# ------------------------------------------------------------
# glimpse() permite revisar rápidamente:
# - número de filas,
# - número de columnas,
# - nombres de variables,
# - tipos de datos,
# - primeros valores observados.
# ------------------------------------------------------------
glimpse(hogares)


# ------------------------------------------------------------
# Se calcula la distribución de la variable objetivo pobre.
#
# count(pobre) cuenta hogares por clase.
# prop = n / sum(n) calcula la proporción relativa de cada clase.
#
# Esto ayuda a detectar si existe desbalance de clases.
# ------------------------------------------------------------
hogares %>%
  count(pobre) %>%
  mutate(prop = n / sum(n))


# ------------------------------------------------------------
# Se divide la base en entrenamiento y prueba.
#
# prop = 0.80 indica que el 80% va a entrenamiento.
# strata = pobre conserva aproximadamente la proporción de pobres
# y no pobres en train y test.
# ------------------------------------------------------------
set.seed(semilla_global)
split_hogares <- initial_split(hogares, prop = 0.80, strata = pobre)


# ------------------------------------------------------------
# Se extraen las bases de entrenamiento y prueba.
# train se usará para ajustar y validar modelos.
# test se reserva para la evaluación final.
# ------------------------------------------------------------
train <- training(split_hogares)
test <- testing(split_hogares)


# ------------------------------------------------------------
# Se construye validación cruzada de 5 folds sobre train.
#
# v = 5 divide el conjunto de entrenamiento en 5 partes.
# strata = pobre mantiene la proporción de clases en cada fold.
# ------------------------------------------------------------
set.seed(semilla_global)
folds_clas <- vfold_cv(train, v = 5, strata = pobre)


# ------------------------------------------------------------
# Se define el conjunto de métricas de clasificación.
#
# f_meas: medida F1, combina precision y recall.
# pr_auc: área bajo la curva precision-recall.
# roc_auc: área bajo la curva ROC.
#
# PR AUC suele ser muy útil cuando la clase de interés
# es relativamente menos frecuente.
# ------------------------------------------------------------
metricas_clas <- metric_set(f_meas, pr_auc, roc_auc)


# ------------------------------------------------------------
# Se muestra un resumen básico de dimensiones:
# - tamaño de train,
# - tamaño de test,
# - número de folds de validación cruzada.
# ------------------------------------------------------------
list(
  train = dim(train),
  test = dim(test),
  folds = length(folds_clas$splits)
)


# ------------------------------------------------------------
# Se define la receta de preprocesamiento para clasificación.
#
# Variable objetivo:
# - pobre
#
# Predictores:
# - región,
# - área,
# - miembros del hogar,
# - edad del jefe,
# - educación del jefe,
# - empleo del jefe,
# - vivienda propia,
# - internet,
# - transferencias.
#
# La receta se estima dentro de los folds para evitar fuga
# de información desde validación o test hacia entrenamiento.
# ------------------------------------------------------------
receta_clas <- recipe(
  pobre ~ region + area + miembros + edad_jefe + educ_jefe + empleo_jefe +
    vivienda_propia + internet + transferencias,
  data = train
) %>%
  # Imputa predictores nominales usando la moda.
  step_impute_mode(all_nominal_predictors()) %>%
  # Imputa predictores numéricos usando la media.
  step_impute_mean(all_numeric_predictors()) %>%
  # Convierte variables categóricas en variables dummy.
  step_dummy(all_nominal_predictors()) %>%
  # Elimina predictores sin variabilidad.
  step_zv(all_predictors()) %>%
  # Normaliza predictores numéricos a media 0 y desviación estándar 1.
  step_normalize(all_numeric_predictors())


# ------------------------------------------------------------
# Se imprime la receta para revisar sus pasos.
# ------------------------------------------------------------
receta_clas


# ------------------------------------------------------------
# Se define un modelo SVM con kernel radial.
#
# La SVM busca una frontera de decisión que separe las clases.
# Con kernel radial puede modelar fronteras no lineales.
#
# cost controla la penalización por errores.
# rbf_sigma controla la escala/localidad del kernel radial.
# ------------------------------------------------------------
# SVM radial: cost penaliza errores; rbf_sigma controla la complejidad/localidad de la frontera.
modelo_svm <- svm_rbf(
  cost = tune(), # mayor cost -> ajuste más estricto al entrenamiento
  rbf_sigma = tune() # mayor sigma -> frontera más local y compleja
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")


# ------------------------------------------------------------
# Se define un modelo k-NN.
#
# k-NN clasifica una observación mirando las clases de sus vecinos
# más cercanos en el espacio de predictores.
#
# neighbors controla cuántos vecinos se usan.
# weight_func controla cómo se ponderan los vecinos.
# dist_power controla la distancia:
# - 1 suele asociarse con Manhattan.
# - 2 suele asociarse con euclídea.
# ------------------------------------------------------------
# k-NN: clasifica usando vecinos cercanos y una regla de distancia.
modelo_knn <- nearest_neighbor(
  neighbors = tune(), # número de vecinos considerados -> mayor fit train
  weight_func = tune(), # pondera igual o más a vecinos cercanos
  dist_power = tune() # 1: Manhattan; 2: euclídea
) %>%
  set_engine("kknn") %>%
  set_mode("classification")


# ------------------------------------------------------------
# Se define un modelo Naive Bayes.
#
# Naive Bayes usa el teorema de Bayes bajo el supuesto simplificador
# de independencia condicional entre predictores.
#
# smoothness y Laplace suavizan probabilidades para evitar problemas
# con eventos raros o categorías poco observadas.
# ------------------------------------------------------------
# Naive Bayes: suaviza probabilidades para evitar estimaciones extremas.
modelo_nb <- naive_Bayes(
  smoothness = tune(), # menor-> más ajuste a datos; mayor-> más suavizado
  Laplace = tune() # mayor Laplace: más suavizado en variables categóricas
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")


# ------------------------------------------------------------
# Se construyen workflows.
#
# Cada workflow combina:
# - la misma receta de preprocesamiento,
# - un algoritmo distinto.
#
# Esto permite comparar modelos bajo condiciones equivalentes.
# ------------------------------------------------------------
# Cada workflow combina la misma receta con un algoritmo distinto.
wf_svm <- workflow() %>% add_recipe(receta_clas) %>% add_model(modelo_svm)
wf_knn <- workflow() %>% add_recipe(receta_clas) %>% add_model(modelo_knn)
wf_nb <- workflow() %>% add_recipe(receta_clas) %>% add_model(modelo_nb)


# ------------------------------------------------------------
# Se muestran los workflows creados.
# ------------------------------------------------------------
list(SVM = wf_svm, kNN = wf_knn, NB = wf_nb)


# ------------------------------------------------------------
# Se define el control para la búsqueda por grilla.
#
# save_pred = TRUE guarda predicciones por fold.
# save_workflow = TRUE conserva el workflow.
# verbose = FALSE evita salida extensa en consola.
# ------------------------------------------------------------
control_grid_lab <- control_grid(
  save_pred = TRUE,
  save_workflow = TRUE,
  verbose = FALSE
)


# ------------------------------------------------------------
# Grilla de hiperparámetros para SVM radial.
#
# Se probarán todas las combinaciones entre:
# - cost: 0.1, 1, 10
# - rbf_sigma: 0.01, 0.05, 0.10
#
# En total son 3 x 3 = 9 combinaciones.
# ------------------------------------------------------------
grid_svm <- tidyr::expand_grid(
  cost = c(0.1, 1, 10),
  rbf_sigma = c(0.01, 0.05, 0.10)
)


# ------------------------------------------------------------
# Grilla de hiperparámetros para k-NN.
#
# neighbors define el número de vecinos.
# weight_func define cómo se ponderan.
# dist_power define el tipo de distancia.
#
# En total son 4 x 2 x 2 = 16 combinaciones.
# ------------------------------------------------------------
grid_knn <- tidyr::expand_grid(
  neighbors = c(3L, 7L, 15L, 25L),
  weight_func = c("rectangular", "triangular"),
  dist_power = c(1, 2)
)


# ------------------------------------------------------------
# Grilla de hiperparámetros para Naive Bayes.
#
# smoothness controla suavizado numérico.
# Laplace controla suavizado para probabilidades categóricas.
#
# En total son 3 x 2 = 6 combinaciones.
# ------------------------------------------------------------
grid_nb <- tidyr::expand_grid(
  smoothness = c(0.5, 1, 2),
  Laplace = c(0, 1)
)


# ------------------------------------------------------------
# Tuning por grilla para SVM.
#
# tune_grid() evalúa cada combinación de hiperparámetros
# usando los folds definidos y las métricas seleccionadas.
# ------------------------------------------------------------
set.seed(semilla_global)
res_svm <- tune_grid(
  wf_svm,
  resamples = folds_clas,
  grid = grid_svm,
  metrics = metricas_clas,
  control = control_grid_lab
)


# ------------------------------------------------------------
# Tuning por grilla para k-NN.
# ------------------------------------------------------------
set.seed(semilla_global)
res_knn <- tune_grid(
  wf_knn,
  resamples = folds_clas,
  grid = grid_knn,
  metrics = metricas_clas,
  control = control_grid_lab
)


# ------------------------------------------------------------
# Tuning por grilla para Naive Bayes.
# ------------------------------------------------------------
set.seed(semilla_global)
res_nb <- tune_grid(
  wf_nb,
  resamples = folds_clas,
  grid = grid_nb,
  metrics = metricas_clas,
  control = control_grid_lab
)


# ------------------------------------------------------------
# Se recolectan las métricas de validación cruzada para los
# tres modelos base: SVM, k-NN y Naive Bayes.
#
# collect_metrics() convierte los resultados de tune_grid()
# en tablas.
# mutate(modelo = ...) agrega una etiqueta para identificar
# cada algoritmo.
# ------------------------------------------------------------
metricas_modelos_base <- bind_rows(
  collect_metrics(res_svm) %>% mutate(modelo = "SVM"),
  collect_metrics(res_knn) %>% mutate(modelo = "k-NN"),
  collect_metrics(res_nb) %>% mutate(modelo = "Naive Bayes")
)


# ------------------------------------------------------------
# Se guardan las métricas de los modelos base.
# ------------------------------------------------------------
write_csv(metricas_modelos_base, file.path(output_dir, "metricas_svm_knn_nb_cv.csv"))


# ------------------------------------------------------------
# Se muestran las tres mejores combinaciones por modelo
# usando PR AUC como métrica de selección.
#
# group_by(modelo) agrupa por algoritmo.
# slice_max(mean, n = 3) conserva las tres mejores medias.
# ------------------------------------------------------------
metricas_modelos_base %>%
  filter(.metric == "pr_auc") %>%
  group_by(modelo) %>%
  slice_max(mean, n = 3, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(desc(mean))


# ------------------------------------------------------------
# Se define un modelo Random Forest de referencia.
#
# mtry y min_n se van a ajustar.
# trees queda fijo en 500.
#
# importance = "impurity" permite calcular importancia de variables.
# probability = TRUE permite obtener probabilidades de clase.
# ------------------------------------------------------------
modelo_rf_ref <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity", probability = TRUE) %>%
  set_mode("classification")


# ------------------------------------------------------------
# Se define un modelo GBM/XGBoost de referencia.
#
# Es un modelo de boosting: entrena árboles secuenciales,
# donde cada árbol intenta corregir errores previos.
#
# Varios hiperparámetros quedan marcados con tune().
# ------------------------------------------------------------
modelo_gbm_ref <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


# ------------------------------------------------------------
# Se crean workflows para RF y GBM usando la misma receta.
# Así se comparan con SVM, k-NN y Naive Bayes bajo el mismo
# preprocesamiento.
# ------------------------------------------------------------
wf_rf_ref <- workflow() %>% add_recipe(receta_clas) %>% add_model(modelo_rf_ref)
wf_gbm_ref <- workflow() %>% add_recipe(receta_clas) %>% add_model(modelo_gbm_ref)


# ------------------------------------------------------------
# Grilla para Random Forest de referencia.
#
# Se prueban combinaciones de:
# - mtry
# - min_n
# ------------------------------------------------------------
grid_rf_ref <- tidyr::expand_grid(
  mtry = c(3L, 7L, 11L),
  min_n = c(5L, 15L)
)


# ------------------------------------------------------------
# Grilla para GBM/XGBoost de referencia.
#
# Tiene más combinaciones porque GBM posee más hiperparámetros
# sensibles.
# ------------------------------------------------------------
grid_gbm_ref <- tidyr::expand_grid(
  tree_depth = c(1L, 3L),
  learn_rate = c(0.03, 0.08),
  loss_reduction = c(1e-10, 0.01),
  sample_size = c(0.70, 0.90),
  mtry = c(3L, 7L),
  min_n = c(5L, 15L)
)


# ------------------------------------------------------------
# Tuning por grilla para Random Forest de referencia.
# ------------------------------------------------------------
set.seed(semilla_global)
res_rf_ref <- tune_grid(
  wf_rf_ref,
  resamples = folds_clas,
  grid = grid_rf_ref,
  metrics = metricas_clas,
  control = control_grid_lab
)


# ------------------------------------------------------------
# Tuning por grilla para GBM/XGBoost de referencia.
# ------------------------------------------------------------
set.seed(semilla_global)
res_gbm_ref <- tune_grid(
  wf_gbm_ref,
  resamples = folds_clas,
  grid = grid_gbm_ref,
  metrics = metricas_clas,
  control = control_grid_lab
)


# ------------------------------------------------------------
# Se juntan las métricas de todos los modelos candidatos:
# - SVM
# - k-NN
# - Naive Bayes
# - RF
# - GBM
#
# Esto permite hacer una comparación global.
# ------------------------------------------------------------
metricas_todos_cv <- bind_rows(
  metricas_modelos_base,
  collect_metrics(res_rf_ref) %>% mutate(modelo = "RF"),
  collect_metrics(res_gbm_ref) %>% mutate(modelo = "GBM")
)


# ------------------------------------------------------------
# Se extrae el mejor resultado de cada modelo y métrica.
#
# Nota metodológica:
# El código filtra las métricas "f1", "roc_auc" y "pr_auc".
# Se mantiene la línea original sin cambios.
# ------------------------------------------------------------
mejores_cv <- metricas_todos_cv %>%
  filter(.metric %in% c("f1", "roc_auc", "pr_auc")) %>%
  group_by(modelo, .metric) %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  ungroup()


# ------------------------------------------------------------
# Se guardan los mejores resultados de validación cruzada.
# ------------------------------------------------------------
write_csv(mejores_cv, file.path(output_dir, "mejores_modelos_cv.csv"))


# ------------------------------------------------------------
# Se muestran los modelos ordenados por PR AUC.
# Esto ayuda a identificar cuál modelo tiene mejor desempeño
# promedio en validación cruzada.
# ------------------------------------------------------------
mejores_cv %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean))


# ------------------------------------------------------------
# Se grafica la comparación de modelos candidatos.
#
# reorder(modelo, mean) ordena las barras por desempeño.
# geom_errorbar() muestra la incertidumbre usando el error estándar.
# facet_wrap(~ .metric) separa el gráfico por métrica.
# coord_flip() gira los ejes para mejorar legibilidad.
# ------------------------------------------------------------
mejores_cv %>%
  ggplot(aes(x = reorder(modelo, mean), y = mean, fill = modelo)) +
  geom_col(show.legend = FALSE, width = 0.65) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.15) +
  facet_wrap(~ .metric, scales = "free_y") +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    x = NULL,
    y = "Promedio en validación cruzada",
    title = "Comparación de modelos candidatos"
  )


# ------------------------------------------------------------
# Curva de validación para SVM.
#
# Se analiza cómo cambia PR AUC según el valor de cost.
# Para cada valor de cost se conserva el mejor resultado.
# ------------------------------------------------------------
curva_svm_cost <- collect_metrics(res_svm) %>%
  filter(.metric == "pr_auc") %>%
  group_by(cost) %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(modelo = "SVM", parametro = "cost", valor = cost)


# ------------------------------------------------------------
# Curva de validación para k-NN.
#
# Se analiza cómo cambia PR AUC según el número de vecinos.
# ------------------------------------------------------------
curva_knn_k <- collect_metrics(res_knn) %>%
  filter(.metric == "pr_auc") %>%
  group_by(neighbors) %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(modelo = "k-NN", parametro = "neighbors", valor = neighbors)


# ------------------------------------------------------------
# Curva de validación para Naive Bayes.
#
# Se analiza cómo cambia PR AUC según smoothness.
# ------------------------------------------------------------
curva_nb_smooth <- collect_metrics(res_nb) %>%
  filter(.metric == "pr_auc") %>%
  group_by(smoothness) %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(modelo = "Naive Bayes", parametro = "smoothness", valor = smoothness)


# ------------------------------------------------------------
# Se juntan las curvas de validación de los tres modelos base.
# ------------------------------------------------------------
curvas_validacion <- bind_rows(curva_svm_cost, curva_knn_k, curva_nb_smooth)


# ------------------------------------------------------------
# Se grafica la relación entre hiperparámetro y PR AUC.
#
# Cada panel muestra un modelo y un hiperparámetro.
# Esto permite observar si el desempeño mejora, empeora o se estabiliza
# al cambiar el hiperparámetro.
# ------------------------------------------------------------
curvas_validacion %>%
  ggplot(aes(x = valor, y = mean, color = modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(modelo ~ parametro, scales = "free_x") +
  labs(
    x = "Valor del hiperparámetro",
    y = "PR AUC en validación cruzada",
    color = "Modelo",
    title = "Curvas de validación"
  )


# ------------------------------------------------------------
# Se selecciona el mejor conjunto de hiperparámetros para cada modelo,
# usando PR AUC como criterio.
# ------------------------------------------------------------
mejor_svm <- select_best(res_svm, metric = "pr_auc")
mejor_knn <- select_best(res_knn, metric = "pr_auc")
mejor_nb <- select_best(res_nb, metric = "pr_auc")
mejor_rf_ref <- select_best(res_rf_ref, metric = "pr_auc")
mejor_gbm_ref <- select_best(res_gbm_ref, metric = "pr_auc")


# ------------------------------------------------------------
# Se finalizan los workflows.
#
# finalize_workflow() reemplaza los tune() por los valores óptimos
# encontrados en la búsqueda por grilla.
#
# El resultado es una lista de workflows listos para entrenarse.
# ------------------------------------------------------------
wfs_finales <- list(
  SVM = finalize_workflow(wf_svm, mejor_svm),
  `k-NN` = finalize_workflow(wf_knn, mejor_knn),
  `Naive Bayes` = finalize_workflow(wf_nb, mejor_nb),
  RF = finalize_workflow(wf_rf_ref, mejor_rf_ref),
  GBM = finalize_workflow(wf_gbm_ref, mejor_gbm_ref)
)


# ------------------------------------------------------------
# Se selecciona el modelo candidato.
#
# Criterio:
# - tomar los resultados de PR AUC,
# - ordenar de mayor a menor media,
# - en caso de comparación, considerar también std_err,
# - seleccionar la primera fila.
# ------------------------------------------------------------
modelo_candidato <- mejores_cv %>%
  filter(.metric == "pr_auc") %>%
  arrange(desc(mean), std_err) %>%
  slice(1)


# ------------------------------------------------------------
# Se imprime la información del modelo candidato.
# ------------------------------------------------------------
modelo_candidato


# ------------------------------------------------------------
# Se extrae el nombre del modelo candidato y luego se recupera
# su workflow final desde la lista wfs_finales.
# ------------------------------------------------------------
nombre_candidato <- modelo_candidato$modelo
nombre_candidato
wf_candidato <- wfs_finales[[nombre_candidato]]


# ------------------------------------------------------------
# Evaluación final del modelo candidato con last_fit().
#
# last_fit():
# - ajusta el modelo con train,
# - evalúa una sola vez en test,
# - devuelve métricas y predicciones finales.
#
# Esta evaluación es la más honesta porque test no se usó
# durante el tuning.
# ------------------------------------------------------------
set.seed(semilla_global)
fit_candidato <- last_fit(
  wf_candidato,
  split = split_hogares,
  metrics = metricas_clas
)


# ------------------------------------------------------------
# Se extraen las métricas y predicciones finales del modelo candidato.
# ------------------------------------------------------------
metricas_test_candidato <- collect_metrics(fit_candidato)
pred_test_candidato <- collect_predictions(fit_candidato)


# ------------------------------------------------------------
# Se guardan métricas y predicciones del modelo candidato.
# ------------------------------------------------------------
write_csv(metricas_test_candidato, file.path(output_dir, "metricas_test_modelo_candidato.csv"))
write_csv(pred_test_candidato, file.path(output_dir, "predicciones_test_modelo_candidato.csv"))


# ------------------------------------------------------------
# Se muestran las métricas finales en test.
# ------------------------------------------------------------
metricas_test_candidato


# ------------------------------------------------------------
# Se calcula la matriz de confusión del modelo candidato.
#
# truth = pobre: clase observada.
# estimate = .pred_class: clase predicha.
# ------------------------------------------------------------
pred_test_candidato %>%
  conf_mat(truth = pobre, estimate = .pred_class)


# ------------------------------------------------------------
# Función para calcular un punto de curva de aprendizaje.
#
# La curva de aprendizaje compara el desempeño del modelo cuando
# se entrena con fracciones crecientes del conjunto de entrenamiento.
#
# Para cada fracción se calcula:
# - desempeño en entrenamiento,
# - desempeño en validación cruzada.
# ------------------------------------------------------------
# Calcula un punto de la curva de aprendizaje para una fracción del train y un modelo.
calcular_punto_aprendizaje <- function(fraccion, nombre_modelo, wf_final) {
  # Cambiamos la semilla con la fracción para que cada submuestra sea reproducible.
  set.seed(semilla_global + round(fraccion * 100))

  # Tomamos una submuestra estratificada para mantener la proporción de clases.
  train_frac <- train %>%
    group_by(pobre) %>%
    slice_sample(prop = fraccion) %>%
    ungroup()

  # Validamos dentro de la submuestra para estimar desempeño fuera de entrenamiento.
  folds_frac <- vfold_cv(train_frac, v = 3, strata = pobre)
  # tidymodels nombra la probabilidad predicha como .pred_<clase>.
  prob_col_evento <- paste0(".pred_", levels(train_frac$pobre)[1])

  # Ajuste sobre la submuestra completa: mide desempeño en entrenamiento.
  ajuste_train <- fit(wf_final, data = train_frac)

  # ------------------------------------------------------------
  # Se calcula PR AUC en la misma muestra de entrenamiento.
  #
  # Este valor puede ser optimista, porque se evalúa sobre los datos
  # usados para ajustar el modelo.
  # ------------------------------------------------------------
  metrica_train <- predict(ajuste_train, train_frac, type = "prob") %>%
    bind_cols(train_frac %>% select(pobre)) %>%
    summarise(
      mean = pr_auc_vec(
        truth = pobre,
        estimate = .data[[prob_col_evento]],
        event_level = "first"
      ),
      std_err = NA_real_
    ) %>%
    mutate(muestra = "Entrenamiento")

  # Reajuste por folds: mide desempeño de validación para la misma fracción.
  metrica_validacion <- fit_resamples(
    wf_final,
    resamples = folds_frac,
    metrics = metricas_clas,
    control = control_resamples(save_pred = FALSE)
  ) %>%
    collect_metrics() %>%
    filter(.metric == "pr_auc") %>%
    select(mean, std_err) %>%
    mutate(muestra = "Validación")

  # Devolvemos ambos puntos con etiquetas para poder graficarlos juntos.
  bind_rows(metrica_train, metrica_validacion) %>%
    mutate(
      fraccion = fraccion,
      n_train = nrow(train_frac),
      modelo = nombre_modelo
    )
}


# ------------------------------------------------------------
# Se definen fracciones crecientes del conjunto de entrenamiento.
#
# Esto permite estudiar si el modelo mejora al recibir más datos.
# ------------------------------------------------------------
# Fracciones crecientes del conjunto de entrenamiento.
fracciones_train <- c(0.25, 0.50, 0.75, 1.00)


# ------------------------------------------------------------
# Se calculan curvas de aprendizaje para todos los modelos finales.
#
# map_dfr() itera sobre fracciones y modelos, y va uniendo
# los resultados en una sola tabla.
# ------------------------------------------------------------
# Calculamos todas las combinaciones de fracción y modelo, y guardamos los resultados.
curvas_aprendizaje <- map_dfr(fracciones_train, function(fraccion) {
  map_dfr(names(wfs_finales), function(modelo) {
    calcular_punto_aprendizaje(fraccion, modelo, wfs_finales[[modelo]])
  })
})


# ------------------------------------------------------------
# Se exportan las curvas de aprendizaje.
# ------------------------------------------------------------
# Exportamos la tabla para poder revisarla fuera del notebook.
write_csv(curvas_aprendizaje, file.path(output_dir, "curvas_aprendizaje.csv"))


# ------------------------------------------------------------
# Se extrae el PR AUC final en test del modelo candidato.
#
# Este valor servirá como referencia horizontal en el gráfico.
# ------------------------------------------------------------
# Referencia de desempeño final en test para el modelo candidato.
test_pr_auc_candidato <- metricas_test_candidato %>%
  filter(.metric == "pr_auc") %>%
  transmute(modelo = nombre_candidato, test_pr_auc = .estimate)


# ------------------------------------------------------------
# Se grafica la curva de aprendizaje.
#
# Eje x: tamaño del entrenamiento.
# Eje y: PR AUC.
# color: modelo.
# linetype: entrenamiento o validación.
#
# La línea punteada representa el desempeño final en test
# del modelo candidato.
# ------------------------------------------------------------
# Gráfico final: compara entrenamiento, validación y test del candidato.
curvas_aprendizaje %>%
  ggplot(aes(x = n_train, y = mean, color = modelo, linetype = muestra)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_errorbar(
    data = curvas_aprendizaje %>% filter(muestra == "Validación"),
    aes(ymin = mean - std_err, ymax = mean + std_err),
    width = 15,
    alpha = 0.6
  ) +
  geom_hline(
    data = test_pr_auc_candidato,
    aes(yintercept = test_pr_auc, color = modelo),
    inherit.aes = FALSE,
    linetype = "dotted",
    linewidth = 0.8
  ) +
  labs(
    x = "Tamaño de entrenamiento",
    y = "PR AUC",
    color = "Modelo",
    linetype = "Muestra",
    title = "Curvas de aprendizaje: training y validación",
    subtitle = "Compara entrenamiento y validación; línea punteada: test del modelo candidato"
  )


# ------------------------------------------------------------
# Se cargan nuevamente los datos cantonales del INEC.
#
# Esta parte abre un segundo bloque de trabajo, ahora orientado
# a una base cantonal con indicadores territoriales.
# ------------------------------------------------------------
data_inec <- suppressMessages(read_excel("data/data_pobreza_INEC.xlsx"))


# ------------------------------------------------------------
# Se prepara una base cantonal para un ejercicio posterior.
#
# canton: nombre del cantón.
# nbi: indicador NBIMEF.
# Las demás variables son predictores territoriales.
#
# nbi_alto define si un cantón pertenece al cuartil superior
# de NBI.
# ------------------------------------------------------------
inec_modelo <- data_inec %>%
  transmute(
    canton = Canton,
    nbi = NBIMEF,
    internet_inst = PorcentajeInstConInternet,
    tef_11_19 = TEF11a19madre,
    cajeros = `CajerosAutomáticosTasapobmay15años`,
    puntos_financieros = `TotPuntosAteFinTasapobmay15años`,
    pib_pc = PIBpercap,
    fecundidad = TasaGlobalFecundmadre,
    participacion_ensenanza = ParticipacionPIBEnseñanza,
    prematuro = Porcentajenacprematuromoderadomujermujermadre,
    agua_potable = aguaPotableViv,
    oficinas = `OficinasTasapobmay15años`,
    nbi_alto = factor(
      if_else(NBIMEF >= quantile(NBIMEF, probs = 0.75, na.rm = TRUE), "NBI alto", "NBI bajo"),
      levels = c("NBI bajo", "NBI alto")
    )
  )


# ------------------------------------------------------------
# Se guarda la base cantonal preparada para el módulo 3.
# ------------------------------------------------------------
write_csv(inec_modelo, file.path(output_dir, "inec_modelo_modulo3.csv"))


# ------------------------------------------------------------
# Se revisa la estructura de la base cantonal.
# ------------------------------------------------------------
glimpse(inec_modelo)


# ------------------------------------------------------------
# Bloque de trabajo para el estudiante.
#
# Aquí no se ejecuta todavía el modelado INEC completo.
# Se dejan instrucciones para que el estudiante replique
# la lógica del laboratorio con la base cantonal.
# ------------------------------------------------------------
# Trabajo del estudiante:
# 1) Crear split_inec, train_inec, test_inec y folds_inec.
# 2) Crear receta_inec_clas excluyendo canton y nbi.
# 3) Reutilizar la lógica de workflows y tuning de este lab.
# 4) Comparar, graficar y seleccionar un modelo candidato.


# ------------------------------------------------------------
# Bloque final opcional para convertir un notebook Jupyter a HTML.
#
# Sys.which("conda") verifica si conda está disponible en el PATH.
#
# Si conda está disponible, se ejecuta nbconvert dentro del ambiente rbase.
# Si no está disponible, se muestra un mensaje y no se intenta convertir.
# ------------------------------------------------------------
if (nzchar(Sys.which("conda"))) {
  system("conda run -n rbase jupyter nbconvert --to html modulo3.ipynb")
} else {
  message("No se ejecutó nbconvert porque conda no está disponible en PATH.")
}
