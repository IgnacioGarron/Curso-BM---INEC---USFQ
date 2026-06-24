# ============================================================
# MÓDULO 2
# Modelos de clasificación y comparación usando tidymodels:
# Random Forest y Gradient Boosting / XGBoost.
#
# La idea general del script es:
# 1. Cargar datos de hogares y datos cantonales del INEC.
# 2. Preparar variables para modelar pobreza.
# 3. Dividir los datos en entrenamiento y prueba.
# 4. Construir recetas de preprocesamiento.
# 5. Entrenar modelos Random Forest y GBM/XGBoost.
# 6. Comparar tuning por grilla y tuning bayesiano.
# 7. Evaluar modelos finales en test.
# 8. Exportar métricas, predicciones y gráficos.
# ============================================================


# ------------------------------------------------------------
# Se define un vector con los paquetes necesarios para el análisis.
# Este vector sirve como referencia de dependencias del script.
# En este código no se instala nada automáticamente, solo se listan.
# ------------------------------------------------------------
paquetes <- c("tidymodels", "tidyverse", "readxl", "ranger", "xgboost")


# ------------------------------------------------------------
# Se cargan los paquetes principales.
# suppressPackageStartupMessages() evita que aparezcan mensajes largos
# de carga de paquetes en la consola, dejando la salida más limpia.
#
# tidymodels: ecosistema para modelado predictivo.
# tidyverse: manipulación, limpieza y visualización de datos.
# readxl: lectura de archivos Excel.
# ------------------------------------------------------------
suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(readxl)
})


# ------------------------------------------------------------
# tidymodels_prefer() resuelve conflictos de nombres de funciones.
# Por ejemplo, si dos paquetes tienen funciones con el mismo nombre,
# prioriza las versiones compatibles con tidymodels.
# ------------------------------------------------------------
tidymodels_prefer()


# ------------------------------------------------------------
# Se fija un tema visual minimalista para los gráficos de ggplot2.
# base_size = 12 controla el tamaño base de las fuentes.
# ------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))


# ------------------------------------------------------------
# Se define una semilla global para reproducibilidad.
# Esto permite que particiones, validaciones cruzadas y búsquedas
# aleatorias produzcan resultados repetibles.
# ------------------------------------------------------------
semilla_global <- 123
set.seed(semilla_global)


# ------------------------------------------------------------
# Se define la carpeta donde se guardarán los resultados del módulo.
# dir.create() crea la carpeta si no existe.
#
# recursive = TRUE permite crear carpetas anidadas.
# showWarnings = FALSE evita advertencias si la carpeta ya existe.
# ------------------------------------------------------------
output_dir <- file.path("output", "modulo2")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# Ruta del archivo de hogares generado o preparado previamente.
# Este archivo se usa como base para los modelos de pobreza a nivel hogar.
# ------------------------------------------------------------
ruta_hogares <- "data/hogares_modulo1.csv"


# ------------------------------------------------------------
# Se leen los datos de hogares desde un CSV.
#
# show_col_types = FALSE evita mostrar el diagnóstico automático
# de tipos de columnas que hace readr::read_csv().
#
# Luego se convierten varias variables categóricas a factor:
# region, area, educ_jefe, empleo_jefe, vivienda_propia e internet.
#
# También se define la variable objetivo pobre como factor ordenado
# con niveles "No pobre" y "Pobre".
# Esto es importante porque las métricas de clasificación necesitan
# saber cuál es la clase positiva.
# ------------------------------------------------------------
hogares <- read_csv(ruta_hogares, show_col_types = FALSE) %>%
  mutate(
    across(c(region, area, educ_jefe, empleo_jefe, vivienda_propia, internet), as.factor),
    pobre = factor(pobre, levels = c("No pobre", "Pobre"))
  )


# ------------------------------------------------------------
# Se cargan datos cantonales reales desde un archivo Excel del INEC.
# suppressMessages() evita mensajes informativos de lectura.
# ------------------------------------------------------------
data_inec <- suppressMessages(read_excel("data/data_pobreza_INEC.xlsx"))


# ------------------------------------------------------------
# Se construye una base resumida para modelado cantonal.
#
# transmute() crea una nueva tabla usando únicamente las variables
# seleccionadas y renombradas.
#
# canton: nombre del cantón.
# nbi: indicador NBIMEF.
# Las demás variables son predictores socioeconómicos o territoriales.
#
# nbi_alto se define como una variable categórica:
# - "NBI alto" si NBIMEF está en el cuartil superior.
# - "NBI bajo" en caso contrario.
#
# quantile(..., probs = .75) calcula el percentil 75.
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
      if_else(NBIMEF >= quantile(NBIMEF, probs = .75, na.rm = TRUE), "NBI alto", "NBI bajo"),
      levels = c("NBI bajo", "NBI alto")
    )
  )


# ------------------------------------------------------------
# Se exportan las bases preparadas.
# Esto permite revisar qué datos entraron al proceso de modelado
# y facilita la reproducibilidad del análisis.
# ------------------------------------------------------------
write_csv(hogares, file.path(output_dir, "hogares_importados.csv"))
write_csv(inec_modelo, file.path(output_dir, "inec_modelo.csv"))


# ------------------------------------------------------------
# glimpse() muestra una vista compacta de las variables:
# número de filas, columnas, nombres, tipos y algunos valores.
# Es útil para verificar rápidamente si la lectura fue correcta.
# ------------------------------------------------------------
glimpse(hogares)
glimpse(inec_modelo)


# ------------------------------------------------------------
# Se divide la base de hogares en entrenamiento y prueba.
#
# prop = 0.80 indica que el 80% de los datos va a entrenamiento
# y el 20% restante a prueba.
#
# strata = pobre mantiene aproximadamente la misma proporción
# de hogares pobres y no pobres en train y test.
# Esto es especialmente importante en clasificación.
# ------------------------------------------------------------
set.seed(semilla_global)
split_hogares <- initial_split(hogares, prop = 0.80, strata = pobre)


# ------------------------------------------------------------
# Se extraen los conjuntos de entrenamiento y prueba
# a partir del objeto split_hogares.
# ------------------------------------------------------------
train <- training(split_hogares)
test <- testing(split_hogares)


# ------------------------------------------------------------
# Partición para modelo de clase.
#
# Se construye validación cruzada de 5 folds sobre los datos de entrenamiento.
# La estratificación por pobre conserva la proporción de clases en cada fold.
# ------------------------------------------------------------
# Partición para modelo de clase
set.seed(semilla_global)
folds_clas <- vfold_cv(train, v = 5, strata = pobre)


# ------------------------------------------------------------
# Partición para modelo de regresión.
#
# Aunque se llama folds_reg, también se estratifica por pobre.
# Esta estructura permitiría evaluar modelos de regresión usando
# la misma lógica de validación cruzada.
# ------------------------------------------------------------
# Partición para modelo de regresión
set.seed(semilla_global)
folds_reg <- vfold_cv(train, v = 5, strata = pobre)


# ------------------------------------------------------------
# Se crea una tabla resumen para comparar train y test.
#
# n: número de observaciones.
# prop_pobre: proporción de hogares pobres.
# ingreso_promedio: ingreso per cápita promedio.
#
# Esto permite verificar que la partición no produjo muestras
# muy diferentes entre entrenamiento y prueba.
# ------------------------------------------------------------
resumen_particion <- tibble(
  muestra = c("train", "test"),
  n = c(nrow(train), nrow(test)),
  prop_pobre = c(mean(train$pobre == "Pobre"), mean(test$pobre == "Pobre")),
  ingreso_promedio = c(mean(train$ingreso_pc), mean(test$ingreso_pc))
)


# ------------------------------------------------------------
# Se define explícitamente el conjunto de predictores usados
# en los modelos de hogares.
#
# Aunque luego las recetas escriben la fórmula completa,
# este vector documenta claramente qué variables explicativas
# se están considerando.
# ------------------------------------------------------------
predictores_hogares <- c(
  "region", "area", "miembros", "edad_jefe", "educ_jefe", "empleo_jefe",
  "vivienda_propia", "internet", "transferencias"
)


# ------------------------------------------------------------
# Receta de preprocesamiento para clasificación.
#
# Objetivo: predecir pobre.
#
# La receta indica:
# - Qué variable se quiere predecir.
# - Qué predictores se usan.
# - Cómo tratar datos faltantes.
# - Cómo convertir variables categóricas.
# - Cómo eliminar variables sin variación.
# - Cómo normalizar variables numéricas.
# ------------------------------------------------------------
receta_clas <- recipe(pobre ~ region + area + miembros + edad_jefe + educ_jefe + empleo_jefe +
                        vivienda_propia + internet + transferencias,
                      data = train
) %>%
  # Imputa variables nominales usando la moda.
  step_impute_mode(all_nominal_predictors()) %>%
  # Imputa variables numéricas usando la media.
  step_impute_mean(all_numeric_predictors()) %>%
  # Convierte variables categóricas en variables dummy.
  step_dummy(all_nominal_predictors()) %>%
  # Elimina predictores con varianza cero.
  step_zv(all_predictors()) %>%
  # Normaliza predictores numéricos para que tengan media 0 y desviación estándar 1.
  step_normalize(all_numeric_predictors())


# ------------------------------------------------------------
# Receta de preprocesamiento para regresión.
#
# Objetivo: predecir ingreso_pc.
#
# Es muy parecida a la receta de clasificación, pero cambia
# la variable dependiente.
# ------------------------------------------------------------
receta_reg <- recipe(ingreso_pc ~ region + area + miembros + edad_jefe + educ_jefe + empleo_jefe +
                       vivienda_propia + internet + transferencias,
                     data = train
) %>%
  # Imputa variables nominales usando la moda.
  step_impute_mode(all_nominal_predictors()) %>%
  # Imputa variables numéricas usando la media.
  step_impute_mean(all_numeric_predictors()) %>%
  # Convierte variables categóricas en variables dummy.
  step_dummy(all_nominal_predictors()) %>%
  # Elimina predictores sin variabilidad.
  step_zv(all_predictors()) %>%
  # Normaliza predictores numéricos.
  step_normalize(all_numeric_predictors())


# ------------------------------------------------------------
# Se define el conjunto de métricas para clasificación.
#
# pr_auc: área bajo la curva precision-recall.
# roc_auc: área bajo la curva ROC.
# f_meas: medida F1, equilibrio entre precision y recall.
#
# PR AUC es especialmente útil cuando hay clases desbalanceadas.
# ------------------------------------------------------------
metricas_clas <- metric_set(
  pr_auc,
  roc_auc,
  f_meas
)


# ------------------------------------------------------------
# Se define el conjunto de métricas para regresión.
#
# rmse: raíz del error cuadrático medio.
# mae: error absoluto medio.
#
# Estas métricas no se usan más adelante en el bloque mostrado,
# pero quedan listas para modelos de regresión.
# ------------------------------------------------------------
metricas_reg <- metric_set(
  rmse,
  mae
)


# ------------------------------------------------------------
# Especificación del modelo Random Forest para clasificación.
#
# mtry = tune() indica que el número de predictores candidatos
# en cada división del árbol será ajustado.
#
# min_n = tune() indica que el tamaño mínimo de nodo también
# será ajustado.
#
# trees = 500 fija el número de árboles del bosque.
#
# engine = "ranger" usa el paquete ranger, eficiente para random forests.
#
# probability = TRUE permite obtener probabilidades de clase,
# necesarias para métricas como roc_auc y pr_auc.
# ------------------------------------------------------------
modelo_rf_clas <- rand_forest(
  mtry = tune(),
  trees = 500, # más árboles reduce la varianza y estabiliza el modelo
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity", probability = TRUE) %>%
  set_mode("classification")


# ------------------------------------------------------------
# Especificación del modelo Gradient Boosting / XGBoost.
#
# boost_tree() define un modelo basado en árboles secuenciales.
# Cada árbol intenta corregir errores de los anteriores.
#
# Los hiperparámetros marcados con tune() serán ajustados.
#
# trees = 500 fija el número de árboles.
# tree_depth controla la profundidad de cada árbol.
# learn_rate controla el tamaño de los pasos de aprendizaje.
# loss_reduction regula divisiones poco útiles.
# sample_size controla la proporción de datos usada por árbol.
# mtry controla cuántas variables se consideran.
# min_n controla el mínimo de observaciones por nodo.
# ------------------------------------------------------------
modelo_gbm_clas <- boost_tree(
  trees = 500, # lo dejamos fijo pero sí se debería tunear
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
# Workflow para Random Forest.
#
# Un workflow une:
# - Una receta de preprocesamiento.
# - Un modelo.
#
# Esto garantiza que el mismo preprocesamiento se aplique
# dentro de cada fold y luego en test, evitando fuga de información.
# ------------------------------------------------------------
wf_rf_clas <- workflow() %>%
  add_recipe(receta_clas) %>%
  add_model(modelo_rf_clas)


# ------------------------------------------------------------
# Workflow para GBM/XGBoost.
# Usa la misma receta de clasificación para que la comparación
# entre modelos sea justa.
# ------------------------------------------------------------
wf_gbm_clas <- workflow() %>%
  add_recipe(receta_clas) %>%
  add_model(modelo_gbm_clas)


# ------------------------------------------------------------
# Se imprimen los workflows para verificar su estructura.
# ------------------------------------------------------------
wf_rf_clas
wf_gbm_clas


# ------------------------------------------------------------
# 1) Control de la búsqueda por grilla.
#
# save_pred = TRUE guarda predicciones de validación cruzada.
# save_workflow = TRUE guarda el workflow usado.
# verbose = FALSE evita salida detallada en consola.
# ------------------------------------------------------------
# 1) Control de la búsqueda por grilla.
#    Guardamos predicciones y workflow para poder diagnosticar resultados después.
control_grid_clas <- control_grid(
  save_pred = TRUE,
  save_workflow = TRUE,
  verbose = FALSE
)


# ------------------------------------------------------------
# 2) Grilla explícita para Random Forest.
#
# expand_grid() genera todas las combinaciones posibles
# entre los valores propuestos de mtry y min_n.
#
# Aquí hay 3 valores de mtry y 3 valores de min_n:
# 3 x 3 = 9 combinaciones.
# ------------------------------------------------------------
# 2) Definimos una grilla explícita para Random Forest.
#    RF solo ajusta mtry y min_n; trees queda fijo en la especificación del modelo.
#    Esta grilla es deliberadamente pequeña para que sea didáctica y ejecutable en clase.
grid_rf_clas <- tidyr::expand_grid(
  mtry = c(3L, 7L, 11L),
  min_n = c(5L, 15L, 25L)
)


# ------------------------------------------------------------
# 3) Grilla explícita para GBM/XGBoost.
#
# Este modelo tiene más hiperparámetros que Random Forest.
# Por eso la grilla crece rápidamente.
#
# Con los valores propuestos:
# 2 x 2 x 2 x 2 x 2 x 2 = 64 combinaciones.
# ------------------------------------------------------------
# 3) Definimos una grilla explícita para GBM/XGBoost.
#    A diferencia de RF, GBM tiene más hiperparámetros sensibles.
#    Incluimos todos los parámetros marcados con tune() en modelo_gbm_clas.
grid_gbm_clas <- tidyr::expand_grid(
  tree_depth = c(1L, 3L),
  learn_rate = c(0.03, 0.08),
  loss_reduction = c(1e-5, 0.01),
  sample_size = c(0.70, 0.90),
  mtry = c(3L, 9L),
  min_n = c(5L, 15L)
)


# ------------------------------------------------------------
# 4) Se resume el tamaño de cada grilla.
#
# Esto ayuda a anticipar el costo computacional antes de ejecutar
# la validación cruzada.
# ------------------------------------------------------------
# 4) Revisamos cuántas combinaciones se evaluarán por modelo.
#    Esto ayuda a anticipar el costo computacional antes de lanzar el tuning.
resumen_grid_clas <- tibble(
  modelo = c("RF", "GBM"),
  combinaciones = c(nrow(grid_rf_clas), nrow(grid_gbm_clas))
)


# ------------------------------------------------------------
# Se imprime el resumen de combinaciones.
# ------------------------------------------------------------
resumen_grid_clas


# ------------------------------------------------------------
# 5) Tuning por grilla para Random Forest.
#
# tune_grid() evalúa cada combinación de hiperparámetros
# usando los folds definidos en folds_clas.
#
# Cada combinación se evalúa con pr_auc, roc_auc y f_meas.
# ------------------------------------------------------------
# 5) Ejecutamos tune_grid() para RF.
#    Cada combinación se evalúa en los mismos folds y con las mismas métricas.
set.seed(semilla_global)
res_rf_grid_clas <- tune_grid(
  wf_rf_clas,
  resamples = folds_clas,
  grid = grid_rf_clas,
  metrics = metricas_clas,
  control = control_grid_clas
)


# ------------------------------------------------------------
# 6) Tuning por grilla para GBM/XGBoost.
#
# Se usa la misma lógica que con RF, pero con la grilla
# específica de GBM.
# ------------------------------------------------------------
# 6) Ejecutamos tune_grid() para GBM.
#    Repetimos la misma estructura para que la comparación RF vs GBM sea simétrica.
set.seed(semilla_global)
res_gbm_grid_clas <- tune_grid(
  wf_gbm_clas,
  resamples = folds_clas,
  grid = grid_gbm_clas,
  metrics = metricas_clas,
  control = control_grid_clas
)


# ------------------------------------------------------------
# 7) Se recolectan las métricas de ambos modelos.
#
# collect_metrics() convierte los resultados de tuning
# en tablas ordenadas.
#
# mutate(modelo = ...) agrega una etiqueta para distinguir RF y GBM.
# mutate(busqueda = "grid") identifica el método de búsqueda usado.
# ------------------------------------------------------------
# 7) Juntamos las métricas de ambos modelos en una sola tabla.
#    La columna busqueda permite distinguir estos resultados de la búsqueda bayesiana.
metricas_grid_clas <- bind_rows(
  collect_metrics(res_rf_grid_clas) %>% mutate(modelo = "RF", busqueda = "grid"),
  collect_metrics(res_gbm_grid_clas) %>% mutate(modelo = "GBM", busqueda = "grid")
) %>%
  select(modelo, busqueda, .metric, .estimator, mean, n, std_err, everything())


# ------------------------------------------------------------
# 8) Se guardan las métricas de la búsqueda por grilla.
# El archivo queda disponible para reportes, auditoría o comparación.
# ------------------------------------------------------------
# 8) Guardamos resultados para revisión posterior o comparación fuera del notebook.
write_csv(metricas_grid_clas, file.path(output_dir, "metricas_grid_clasificacion.csv"))


# ------------------------------------------------------------
# 9) Se seleccionan los mejores hiperparámetros por PR AUC.
#
# select_best() devuelve la combinación con mejor desempeño
# para la métrica indicada.
# ------------------------------------------------------------
# 9) Identificamos el mejor conjunto de hiperparámetros por PR AUC.
#    PR AUC es útil cuando la clase positiva es relativamente menos frecuente.
mejor_rf_grid_clas <- select_best(res_rf_grid_clas, metric = "pr_auc")
mejor_gbm_grid_clas <- select_best(res_gbm_grid_clas, metric = "pr_auc")


# ------------------------------------------------------------
# Se construye una tabla con el mejor resultado de cada modelo
# según PR AUC.
#
# show_best(..., n = 1) muestra la mejor combinación encontrada.
# ------------------------------------------------------------
mejores_grid_clas <- bind_rows(
  show_best(res_rf_grid_clas, metric = "pr_auc", n = 1) %>% mutate(modelo = "RF"),
  show_best(res_gbm_grid_clas, metric = "pr_auc", n = 1) %>% mutate(modelo = "GBM")
) %>%
  select(modelo, everything())


# ------------------------------------------------------------
# 10) Se muestra la comparación compacta de mejores modelos
# bajo búsqueda por grilla.
# ------------------------------------------------------------
# 10) Mostramos una tabla compacta para comparar RF y GBM bajo tune_grid().
mejores_grid_clas


# ------------------------------------------------------------
# 1) Control de la búsqueda bayesiana.
#
# La búsqueda bayesiana no prueba todas las combinaciones posibles.
# En cambio, usa los resultados previos para proponer nuevas zonas
# prometedoras del espacio de hiperparámetros.
# ------------------------------------------------------------
# 1) Control de la búsqueda bayesiana.
set.seed(semilla_global)


# ------------------------------------------------------------
# no_improve detiene el proceso si no hay mejora durante varias iteraciones.
# save_pred = TRUE guarda predicciones.
# save_workflow = TRUE guarda el workflow.
# verbose = FALSE reduce mensajes en consola.
# ------------------------------------------------------------
#    no_improve detiene el proceso si varias iteraciones no mejoran la métrica elegida.
#    Guardamos el workflow, pero no las predicciones, para reducir memoria en la fase iterativa.
control_bayes_clas <- control_bayes(
  save_pred = TRUE,
  save_workflow = TRUE,
  no_improve = 10,
  verbose = FALSE
)


# ------------------------------------------------------------
# 2) Espacio de búsqueda bayesiano para Random Forest.
#
# A diferencia de una grilla fija, aquí se define un rango.
# El algoritmo bayesiano puede explorar valores dentro de ese rango.
# ------------------------------------------------------------
# 2) Definimos el espacio de búsqueda para RF.
#    Debe cubrir los hiperparámetros evaluados en tune_grid() y permitir explorar alrededor.
params_rf_bayes_clas <- parameters(
  mtry(range = c(2L, 12L)),
  min_n(range = c(2L, 50L))
)


# ------------------------------------------------------------
# 3) Espacio de búsqueda bayesiano para GBM/XGBoost.
#
# Se definen rangos para todos los hiperparámetros marcados
# con tune() en el modelo GBM.
#
# En tidymodels/dials algunos parámetros se manejan en escala transformada,
# por ejemplo learn_rate y loss_reduction.
# ------------------------------------------------------------
# 3) Definimos el espacio de búsqueda para GBM.
#    Incluimos todos los hiperparámetros tuneados en modelo_gbm_clas.
#    learn_rate usa escala log10 en dials; por eso el rango va de 10^-4 a 10^-1.
params_gbm_bayes_clas <- parameters(
  tree_depth(range = c(1L, 6L)),
  learn_rate(range = c(-4, -1)),
  loss_reduction(range = c(-10, 0)),
  sample_prop(range = c(0.50, 1.00)),
  mtry(range = c(2L, 12L)),
  min_n(range = c(2L, 40L))
)


# ------------------------------------------------------------
# 4) Objetos de inicio para búsqueda bayesiana.
#
# Estos objetos guardan los resultados obtenidos por grilla.
# Aunque en las llamadas posteriores el código usa initial = 20,
# estos objetos documentan la intención de conectar grid y Bayes.
# ------------------------------------------------------------
# 4) Usamos como initial los resultados de tune_grid() del punto 4.1.
#    Esto evita repetir tune_grid() y conecta explícitamente ambos métodos.
#    tune_bayes() aprovecha todos los puntos evaluados en la grilla, incluyendo el mejor.
inicio_rf_bayes_clas <- res_rf_grid_clas
inicio_gbm_bayes_clas <- res_gbm_grid_clas


# ------------------------------------------------------------
# 5) Búsqueda bayesiana para Random Forest.
#
# tune_bayes() busca hiperparámetros de manera más inteligente
# que una grilla exhaustiva.
#
# initial = 20 indica cuántos puntos iniciales se usan.
# iter = 20 indica cuántas iteraciones bayesianas adicionales se ejecutan.
# param_info define el espacio de búsqueda.
# ------------------------------------------------------------
# 5) Ejecutamos tune_bayes() para RF.
#    A partir de la grilla previa, Bayes propone nuevas combinaciones prometedoras.
set.seed(semilla_global)
res_rf_bayes_clas <- tune_bayes(
  wf_rf_clas,
  resamples = folds_clas,
  initial = 20, # inicio_rf_bayes_clas,
  iter = 20, # número de intentos bayesianos
  metrics = metricas_clas,
  control = control_bayes_clas,
  param_info = params_rf_bayes_clas # espacio de búsqueda
)


# ------------------------------------------------------------
# 6) Búsqueda bayesiana para GBM/XGBoost.
#
# Se aplica la misma estrategia que en RF para mantener
# una comparación metodológica ordenada.
# ------------------------------------------------------------
# 6) Ejecutamos tune_bayes() para GBM.
#    Mantener la misma lógica facilita comparar RF vs GBM y grid vs Bayes.
set.seed(semilla_global)
res_gbm_bayes_clas <- tune_bayes(
  wf_gbm_clas,
  resamples = folds_clas,
  initial = 20, # inicio_gbm_bayes_clas,
  iter = 20, # número de intentos bayesianos
  metrics = metricas_clas,
  control = control_bayes_clas,
  param_info = params_gbm_bayes_clas
)


# ------------------------------------------------------------
# 7) Se recolectan métricas de la búsqueda bayesiana.
#
# Igual que antes, se etiquetan los resultados por modelo y por tipo
# de búsqueda para poder compararlos luego.
# ------------------------------------------------------------
# 7) Reunimos las métricas bayesianas de ambos modelos.
metricas_bayes_clas <- bind_rows(
  collect_metrics(res_rf_bayes_clas) %>% mutate(modelo = "RF", busqueda = "bayes"),
  collect_metrics(res_gbm_bayes_clas) %>% mutate(modelo = "GBM", busqueda = "bayes")
) %>%
  select(modelo, busqueda, .metric, .estimator, mean, n, std_err, everything())


# ------------------------------------------------------------
# 8) Se guardan las métricas bayesianas en CSV.
# Esto permite comparar posteriormente grid vs bayes.
# ------------------------------------------------------------
# 8) Guardamos resultados para comparar contra tune_grid().
write_csv(metricas_bayes_clas, file.path(output_dir, "metricas_bayes_clasificacion.csv"))


# ------------------------------------------------------------
# 9) Se seleccionan los mejores hiperparámetros bayesianos
# usando PR AUC como criterio.
# ------------------------------------------------------------
# 9) Seleccionamos los mejores hiperparámetros por PR AUC.
mejor_rf_bayes_clas <- select_best(res_rf_bayes_clas, metric = "pr_auc")
mejor_gbm_bayes_clas <- select_best(res_gbm_bayes_clas, metric = "pr_auc")


# ------------------------------------------------------------
# Se crea una tabla compacta con el mejor resultado de RF y GBM
# bajo búsqueda bayesiana.
# ------------------------------------------------------------
mejores_bayes_clas <- bind_rows(
  show_best(res_rf_bayes_clas, metric = "pr_auc", n = 1) %>% mutate(modelo = "RF"),
  show_best(res_gbm_bayes_clas, metric = "pr_auc", n = 1) %>% mutate(modelo = "GBM")
) %>%
  select(modelo, everything())


# ------------------------------------------------------------
# 10) Se muestran los mejores resultados de la búsqueda bayesiana.
# La segunda línea repite la impresión del mismo objeto.
# ------------------------------------------------------------
# 10) Mostramos el mejor resultado encontrado por la búsqueda bayesiana.
mejores_bayes_clas
mejores_bayes_clas


# ------------------------------------------------------------
# Se comparan los mejores resultados de grid y Bayes.
#
# bind_cols() agrega la columna tune.
# bind_rows() une ambos enfoques.
# arrange(desc(mean)) ordena de mayor a menor desempeño promedio.
# ------------------------------------------------------------
bind_rows(
  bind_cols(mejores_grid_clas, tune = "grid"),
  bind_cols(mejores_bayes_clas, tune = "bayes")
) %>%
  select(tune, modelo, mean, std_err) %>%
  arrange(desc(mean))


# ------------------------------------------------------------
# Se eligen los mejores hiperparámetros finales para cada modelo.
# Aquí se toma la búsqueda bayesiana como referencia final.
# ------------------------------------------------------------
mejor_rf_clas <- select_best(res_rf_bayes_clas, metric = "pr_auc")
mejor_gbm_clas <- select_best(res_gbm_bayes_clas, metric = "pr_auc")


# ------------------------------------------------------------
# finalize_workflow() inserta los hiperparámetros seleccionados
# dentro del workflow.
#
# A partir de aquí, los modelos ya no tienen parámetros tune().
# Están listos para entrenamiento final.
# ------------------------------------------------------------
final_rf_clas <- finalize_workflow(wf_rf_clas, mejor_rf_clas)
final_gbm_clas <- finalize_workflow(wf_gbm_clas, mejor_gbm_clas)


# ------------------------------------------------------------
# last_fit() entrena el modelo final con los datos de entrenamiento
# y lo evalúa una sola vez en test.
#
# Esta es la evaluación final honesta, porque el test no se usó
# durante la selección de hiperparámetros.
# ------------------------------------------------------------
set.seed(semilla_global)
fit_rf_clas <- last_fit(final_rf_clas, split = split_hogares, metrics = metricas_clas)


# ------------------------------------------------------------
# Evaluación final del modelo GBM/XGBoost sobre el mismo split.
# ------------------------------------------------------------
set.seed(semilla_global)
fit_gbm_clas <- last_fit(final_gbm_clas, split = split_hogares, metrics = metricas_clas)


# ------------------------------------------------------------
# Se recolectan las métricas finales en test para ambos modelos.
#
# collect_metrics() extrae las métricas calculadas por last_fit().
# mutate(modelo = ...) identifica de qué modelo viene cada fila.
# ------------------------------------------------------------
metricas_test_clas <- bind_rows(
  collect_metrics(fit_rf_clas) %>% mutate(modelo = "RF"),
  collect_metrics(fit_gbm_clas) %>% mutate(modelo = "GBM")
) %>%
  select(modelo, .metric, .estimator, .estimate)


# ------------------------------------------------------------
# Se recolectan las predicciones finales en test.
#
# Esto permite construir matrices de confusión, curvas ROC,
# curvas PR o revisar casos individuales.
# ------------------------------------------------------------
pred_test_clas <- bind_rows(
  collect_predictions(fit_rf_clas) %>% mutate(modelo = "RF"),
  collect_predictions(fit_gbm_clas) %>% mutate(modelo = "GBM")
)


# ------------------------------------------------------------
# Se calculan matrices de confusión por modelo.
#
# group_by(modelo) permite obtener una matriz separada para RF y GBM.
# truth = pobre indica la clase real.
# estimate = .pred_class indica la clase predicha.
# ------------------------------------------------------------
matrices_confusion <- pred_test_clas %>%
  group_by(modelo) %>%
  conf_mat(truth = pobre, estimate = .pred_class)


# ------------------------------------------------------------
# Se guardan métricas y predicciones finales en archivos CSV.
# Esto deja evidencia del desempeño en test.
# ------------------------------------------------------------
write_csv(metricas_test_clas, file.path(output_dir, "metricas_test_clasificacion.csv"))
write_csv(pred_test_clas, file.path(output_dir, "predicciones_test_clasificacion.csv"))


# ------------------------------------------------------------
# Se muestran las métricas finales de test.
# ------------------------------------------------------------
metricas_test_clas


# ------------------------------------------------------------
# Comparación entre validación cruzada y test.
#
# Primero se extraen los mejores resultados de validación cruzada
# obtenidos durante tune_bayes().
#
# Para cada modelo y cada métrica, se conserva la mejor media.
# ------------------------------------------------------------
# Mejores resultados de validación cruzada obtenidos con tune_bayes().
comparacion_cv_clas <- metricas_bayes_clas %>%
  filter(.metric %in% c("f_meas", "roc_auc", "pr_auc")) %>%
  group_by(modelo, .metric) %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    modelo,
    muestra = "Validación cruzada",
    .metric,
    .estimate = mean
  )


# ------------------------------------------------------------
# Se preparan los resultados finales en test calculados con last_fit().
# Se usan las mismas métricas para poder compararlas contra CV.
# ------------------------------------------------------------
# Resultados finales en test calculados con last_fit().
comparacion_test_clas <- metricas_test_clas %>%
  filter(.metric %in% c("f_meas", "roc_auc", "pr_auc")) %>%
  transmute(
    modelo,
    muestra = "Test",
    .metric,
    .estimate
  )


# ------------------------------------------------------------
# Se unen los resultados de validación cruzada y test.
# Esta tabla permite observar si el desempeño cae mucho al pasar
# de validación a datos no vistos.
# ------------------------------------------------------------
comparacion_clas <- bind_rows(comparacion_cv_clas, comparacion_test_clas)


# ------------------------------------------------------------
# Se guarda la comparación CV vs test.
# ------------------------------------------------------------
write_csv(comparacion_clas, file.path(output_dir, "comparacion_cv_test_clasificacion.csv"))


# ------------------------------------------------------------
# Se grafica la comparación entre validación cruzada y test.
#
# Eje x: tipo de muestra.
# Eje y: valor de la métrica.
# fill: modelo.
# facet_wrap(): separa el gráfico por métrica.
#
# El objetivo es ver si los modelos mantienen desempeño
# fuera de la validación cruzada.
# ------------------------------------------------------------
comparacion_clas %>%
  ggplot(aes(x = muestra, y = .estimate, fill = modelo)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  facet_wrap(~ .metric, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    x = NULL,
    y = "Valor de la métrica",
    fill = "Modelo",
    title = "Validación cruzada vs. test"
  )


# ------------------------------------------------------------
# Bloque final opcional para convertir un notebook Jupyter a HTML.
#
# Sys.which("conda") verifica si conda está disponible en el PATH.
#
# Si conda existe, ejecuta nbconvert dentro del ambiente rbase.
# Si no existe, muestra un mensaje y no intenta convertir el notebook.
# ------------------------------------------------------------
if (nzchar(Sys.which("conda"))) {
  system("conda run -n rbase jupyter nbconvert --to html modulo2.ipynb")
} else {
  message("No se ejecutó nbconvert porque conda no está disponible en PATH.")
}
