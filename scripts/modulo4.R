# ==============================================================================
# MODULO 4: PCA, CLUSTERING EXPLORATORIO Y CIERRE DEL FLUJO DE MACHINE LEARNING
# ==============================================================================
#
# Este script reproduce todo el codigo R del notebook modulo4.ipynb.
#
# Objetivos:
#   1. Preparar indicadores de movilidad cantonal para analisis no supervisado.
#   2. Reducir su dimensionalidad mediante componentes principales (PCA).
#   3. Interpretar la varianza explicada y las cargas del PCA.
#   4. Segmentar cantones mediante k-means.
#   5. Evaluar distintos numeros de clusters con metricas internas.
#   6. Describir los perfiles de movilidad de los clusters obtenidos.
#   7. Preparar la base INEC para repetir el ejercicio con datos socioeconomicos.
#
# En aprendizaje no supervisado no existe una variable objetivo que indique la
# respuesta correcta. Por eso la evaluacion combina metricas internas,
# visualizaciones y la interpretabilidad sustantiva de los grupos.

# ==============================================================================
# 0. PAQUETES, SEMILLA Y DIRECTORIO DE SALIDA
# ==============================================================================

# Paquetes utilizados:
# - tidymodels: recetas y workflows.
# - tidyverse: manipulacion, lectura y visualizacion de datos.
# - tidyclust: modelos y metricas de clustering compatibles con tidymodels.
# - broom: convierte resultados de modelos en tablas ordenadas.
# - readxl: importa la base alternativa de Excel del INEC.
paquetes <- c("tidymodels", "tidyverse", "tidyclust", "broom", "readxl")

# Esta comprobacion no instala paquetes automaticamente. Solo informa cuales
# faltan para que el usuario pueda decidir cuando instalarlos.
paquetes_faltantes <- paquetes[
  !vapply(paquetes, requireNamespace, logical(1), quietly = TRUE)
]
paquetes_faltantes

# Si aparece algun paquete faltante, se puede instalar con:
# install.packages(paquetes_faltantes)

# Cargamos los paquetes sin mostrar los mensajes habituales de inicio.
suppressPackageStartupMessages({
  library(tidymodels)
  library(tidyverse)
  library(tidyclust)
  library(broom)
  library(readxl)
})

# Da preferencia a las funciones de tidymodels cuando existen nombres
# compartidos con otros paquetes.
tidymodels_prefer()

# Establece un tema grafico comun para todas las visualizaciones del modulo.
theme_set(theme_minimal(base_size = 12))

# La semilla hace reproducibles las particiones y los ajustes aleatorios.
semilla_global <- 123
set.seed(semilla_global)

# Todas las tablas producidas por el modulo se guardan en esta carpeta.
output_dir <- file.path("output", "modulo4")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# 1. IMPORTACION Y PREPARACION DE LOS DATOS DE MOVILIDAD
# ==============================================================================

# La base contiene promedios de movilidad por canton, provincia, dia de la
# semana y franja horaria.
roads <- read_csv("data/roads_canton_mean.csv", show_col_types = FALSE) %>%
  mutate(
    # Provincia se conserva como variable categorica para interpretar los
    # clusters, pero no se usara para construirlos.
    provincia = as.factor(provincia),

    # Proporcion de movilidad correspondiente al fin de semana.
    prop_weekend = weekend_all / (weekday_all + weekend_all),

    # Proporcion de movilidad que ocurre durante la noche.
    prop_night = (weekday_night + weekend_night) /
      (weekday_all + weekend_all)
  )

# Inspeccionamos tipos de variables y primeras observaciones.
glimpse(roads)

# Resumen descriptivo general de la cobertura y volumen de la base.
roads %>%
  summarise(
    cantones = n(),
    provincias = n_distinct(provincia),
    viajes_promedio = mean(viajes, na.rm = TRUE),
    viajeros_promedio = mean(viajeros, na.rm = TRUE)
  )

# ==============================================================================
# 2. MATRIZ NUMERICA PARA EL ANALISIS NO SUPERVISADO
# ==============================================================================

# PCA y k-means dependen de distancias y varianzas. Por eso las variables deben
# quedar en una escala comparable. La receta:
#   - imputa faltantes numericos con la media;
#   - elimina predictores sin variacion;
#   - centra y estandariza las variables.
#
# No se incluyen identificadores, provincia ni totales generales. Queremos que
# la estructura se base en patrones temporales de movilidad.
receta_unsup <- recipe(
  ~ monday + tuesday + wednesday + thursday + friday + saturday + sunday +
    weekday_morning + weekday_afternoon + weekday_evening + weekday_night +
    weekend_morning + weekend_afternoon + weekend_evening + weekend_night +
    prop_weekend + prop_night,
  data = roads
) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# prep() aprende los parametros de preprocesamiento, como medias y desviaciones.
prep_unsup <- prep(receta_unsup)

# bake() aplica la receta aprendida a todas las observaciones.
x_roads <- bake(prep_unsup, new_data = roads) %>%
  as_tibble()

# Estas variables no participan en PCA ni clustering. Se reservan para
# interpretar posteriormente las posiciones y los grupos de cada canton.
metadata_roads <- roads %>%
  select(
    canton,
    name_es,
    provincia,
    viajes,
    viajeros,
    weekday_all,
    weekend_all,
    prop_weekend,
    prop_night
  )

# Verificamos el numero de cantones y variables numericas procesadas.
list(
  filas = nrow(x_roads),
  columnas = ncol(x_roads)
)

# ==============================================================================
# 3. ANALISIS DE COMPONENTES PRINCIPALES (PCA)
# ==============================================================================

# prcomp() calcula combinaciones lineales ortogonales de las variables.
# No volvemos a centrar ni escalar porque la receta ya realizo ambos pasos.
pca_roads <- prcomp(x_roads, center = FALSE, scale. = FALSE)

# Extraemos la desviacion, varianza y proporcion explicada por componente.
varianza_pca <- tidy(pca_roads, matrix = "eigenvalues") %>%
  transmute(
    componente = paste0("PC", PC),
    num_componente = PC,
    desviacion_estandar = std.dev,
    varianza = std.dev^2,
    prop_varianza = percent,
    prop_acumulada = cumulative
  ) %>%
  mutate(across(c(prop_varianza, prop_acumulada), as.numeric))

# Exportamos la tabla para poder revisar la varianza fuera de R.
write_csv(
  varianza_pca,
  file.path(output_dir, "varianza_pca_roads.csv")
)

# Mostramos las primeras diez componentes.
varianza_pca %>%
  slice(1:10)

# Grafico tipo scree:
# - las barras representan la varianza individual;
# - la linea representa la varianza acumulada.
varianza_pca %>%
  slice(1:10) %>%
  ggplot(aes(x = num_componente, y = prop_varianza)) +
  geom_col(fill = "#4C78A8", width = 0.7) +
  geom_line(
    aes(y = prop_acumulada),
    color = "#F58518",
    linewidth = 0.8
  ) +
  geom_point(
    aes(y = prop_acumulada),
    color = "#F58518",
    size = 2
  ) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Componente principal",
    y = "Proporcion de varianza",
    title = "Varianza explicada por PCA"
  )

# ------------------------------------------------------------------------------
# 3.1 Scores: posicion de cada canton en el espacio PCA
# ------------------------------------------------------------------------------

# Los scores son las coordenadas de cada canton en las nuevas componentes.
# Conservamos las primeras cuatro y luego unimos la metadata descriptiva.
scores_pca <- tidy(pca_roads, matrix = "scores") %>%
  filter(PC <= 4) %>%
  mutate(componente = paste0("PC", PC)) %>%
  select(row, componente, value) %>%
  pivot_wider(names_from = componente, values_from = value) %>%
  arrange(row) %>%
  select(-row) %>%
  bind_cols(metadata_roads)

write_csv(
  scores_pca,
  file.path(output_dir, "scores_pca_roads.csv")
)

# PC1-PC2 permite observar estructura, separacion o solapamiento. El color y
# tamano aportan contexto, pero no intervienen en el PCA.
scores_pca %>%
  ggplot(aes(
    x = PC1,
    y = PC2,
    color = prop_weekend,
    size = viajes
  )) +
  geom_point(alpha = 0.75) +
  scale_color_viridis_c(option = "C") +
  labs(
    x = "PC1",
    y = "PC2",
    color = "Proporcion fin de semana",
    size = "Viajes",
    title = "Cantones proyectados en PC1-PC2"
  )

# ------------------------------------------------------------------------------
# 3.2 Cargas: contribucion de las variables originales a cada componente
# ------------------------------------------------------------------------------

# Una carga alta en valor absoluto indica que una variable contribuye mucho a
# definir una componente. El signo indica la direccion de la asociacion.
cargas_pca <- tidy(pca_roads, matrix = "loadings") %>%
  filter(PC <= 3) %>%
  transmute(
    variable = column,
    componente = paste0("PC", PC),
    carga = value
  ) %>%
  mutate(abs_carga = abs(carga))

write_csv(
  cargas_pca,
  file.path(output_dir, "cargas_pca_roads.csv")
)

# Identificamos las ocho variables con mayor carga absoluta por componente.
cargas_pca %>%
  group_by(componente) %>%
  slice_max(abs_carga, n = 8, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(componente, desc(abs_carga))

# Visualizamos las variables que mas definen PC1 y PC2.
cargas_pca %>%
  filter(componente %in% c("PC1", "PC2")) %>%
  group_by(componente) %>%
  slice_max(abs_carga, n = 8, with_ties = FALSE) %>%
  ungroup() %>%
  ggplot(aes(
    x = reorder(variable, abs_carga),
    y = carga,
    fill = componente
  )) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ componente, scales = "free_y") +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    x = NULL,
    y = "Carga",
    title = "Variables con mayor peso en PC1 y PC2"
  )

# ==============================================================================
# 4. K-MEANS SOBRE COMPONENTES PRINCIPALES
# ==============================================================================

# El PCA se incluye dentro de la receta del workflow. De este modo, durante la
# validacion cruzada, las componentes se aprenden exclusivamente con la parte de
# entrenamiento de cada fold. Esto evita fuga de informacion.

# Retenemos las componentes necesarias para explicar al menos el 90% de la
# varianza de los indicadores originales.
umbral_varianza <- 0.90

# Matriz utilizada especificamente para clustering.
datos_cluster <- roads %>%
  select(
    monday,
    tuesday,
    wednesday,
    thursday,
    friday,
    saturday,
    sunday,
    weekday_morning,
    weekday_afternoon,
    weekday_evening,
    weekday_night,
    weekend_morning,
    weekend_afternoon,
    weekend_evening,
    weekend_night,
    prop_weekend,
    prop_night
  )

# La receta aprende imputacion, normalizacion y PCA en un solo flujo.
receta_cluster <- recipe(~ ., data = datos_cluster) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(
    all_numeric(),
    threshold = umbral_varianza,
    id = "pca_cluster"
  )

# Esta preparacion se usa solo para inspeccionar cuantas componentes retiene el
# umbral. El workflow volvera a estimar la receta dentro de cada fold.
prep_cluster_info <- prep(receta_cluster)
x_pca_cluster <- bake(prep_cluster_info, new_data = datos_cluster)

n_componentes <- x_pca_cluster %>%
  select(starts_with("PC")) %>%
  ncol()

list(
  umbral_varianza = umbral_varianza,
  n_componentes = n_componentes,
  columnas_cluster = ncol(datos_cluster)
)

# ------------------------------------------------------------------------------
# 4.1 Evaluacion de distintos numeros de clusters
# ------------------------------------------------------------------------------

# Creamos cinco folds. Como no existe variable objetivo, no se estratifica.
set.seed(semilla_global)
folds_cluster <- vfold_cv(datos_cluster, v = 5)

# num_clusters = tune() indica que k sera seleccionado por tuning.
modelo_kmeans <- k_means(num_clusters = tune()) %>%
  set_engine("stats")

# El workflow encapsula la preparacion y el modelo de clustering.
wf_kmeans <- workflow() %>%
  add_recipe(receta_cluster) %>%
  add_model(modelo_kmeans)

# Valores candidatos de k.
grid_k <- tibble(num_clusters = 2:8)

# tune_cluster() evalua cada k en los mismos folds.
#
# Metricas:
# - sse_within_total: suma de cuadrados dentro de los clusters.
# - sse_total: variacion total de los datos.
# - sse_ratio: proporcion de variacion que permanece dentro de los clusters.
res_kmeans <- tune_cluster(
  wf_kmeans,
  resamples = folds_cluster,
  grid = grid_k,
  control = control_grid(
    save_pred = TRUE,
    extract = identity
  ),
  metrics = cluster_metric_set(
    sse_within_total,
    sse_total,
    sse_ratio
  )
)

# Reunimos las metricas promedio y su error estandar.
resultados_k <- collect_metrics(res_kmeans)

write_csv(
  resultados_k,
  file.path(output_dir, "evaluacion_kmeans_roads.csv")
)

resultados_k

# ------------------------------------------------------------------------------
# 4.2 Criterio exploratorio del codo
# ------------------------------------------------------------------------------

# sse_ratio suele disminuir cuando k aumenta. No elegimos simplemente el menor
# valor, porque eso favoreceria demasiados clusters. Buscamos el punto donde las
# mejoras adicionales comienzan a ser pequenas.
resultados_k %>%
  filter(.metric == "sse_ratio") %>%
  ggplot(aes(x = num_clusters, y = mean)) +
  geom_line(linewidth = 0.8, color = "#4C78A8") +
  geom_point(size = 2, color = "#F58518") +
  geom_errorbar(
    aes(
      ymin = mean - std_err,
      ymax = mean + std_err
    ),
    width = 0.15,
    alpha = 0.6
  ) +
  scale_x_continuous(breaks = 2:8) +
  labs(
    x = "Numero de clusters k",
    y = "SSE dentro / SSE total",
    title = "Criterio de codo para elegir k"
  )

# Tabla detallada de metricas para comparar los valores candidatos.
resultados_k %>%
  filter(.metric %in% c(
    "sse_within_total",
    "sse_total",
    "sse_ratio"
  )) %>%
  select(num_clusters, .metric, mean, std_err) %>%
  arrange(.metric, num_clusters)

# ------------------------------------------------------------------------------
# 4.3 Seleccion automatica de k y ajuste final
# ------------------------------------------------------------------------------

# Calculamos la mejora relativa de sse_ratio al pasar de k - 1 a k.
mejora_k <- resultados_k %>%
  filter(.metric == "sse_ratio") %>%
  arrange(num_clusters) %>%
  mutate(
    mejora_relativa = (
      dplyr::lag(mean) - mean
    ) / dplyr::lag(mean),
    mejora_relativa = replace_na(mejora_relativa, Inf)
  )

# Seleccionamos el primer k cuya mejora relativa sea menor al 50%.
# Este umbral es una regla didactica y debe complementarse con interpretacion.
umbral <- 0.50

k_candidato <- mejora_k %>%
  filter(
    num_clusters > min(num_clusters),
    mejora_relativa < umbral
  ) %>%
  slice(1) %>%
  pull(num_clusters)

# Si ninguna mejora cae por debajo del umbral, usamos el k con menor sse_ratio.
if (length(k_candidato) == 0) {
  k_candidato <- mejora_k %>%
    slice_min(mean, n = 1, with_ties = FALSE) %>%
    pull(num_clusters)
}

# Sustituimos el parametro tune() por el k seleccionado.
final_kmeans <- finalize_workflow(
  wf_kmeans,
  tibble(num_clusters = k_candidato)
)

# Ajustamos la receta y k-means sobre todos los datos disponibles.
set.seed(semilla_global)
fit_kmeans_final <- fit(final_kmeans, data = datos_cluster)

# Extraemos la receta ya estimada para obtener las coordenadas PCA usadas por el
# modelo final y agregamos la metadata descriptiva de cada canton.
scores_cluster <- extract_recipe(
  fit_kmeans_final,
  estimated = TRUE
) %>%
  bake(new_data = datos_cluster) %>%
  bind_cols(metadata_roads)

# predict() asigna a cada observacion el centroide mas cercano.
clusters_roads <- scores_cluster %>%
  bind_cols(
    predict(fit_kmeans_final, new_data = datos_cluster)
  ) %>%
  mutate(cluster = factor(.pred_cluster)) %>%
  select(-.pred_cluster)

write_csv(
  clusters_roads,
  file.path(output_dir, "clusters_roads.csv")
)

# Distribucion de cantones entre clusters.
clusters_roads %>%
  count(cluster) %>%
  mutate(prop = n / sum(n))

# Visualizamos los clusters en PC1-PC2 cuando existen al menos dos componentes.
# Si solo existe PC1, usamos jitter para evitar que los puntos se superpongan.
if ("PC2" %in% names(clusters_roads)) {
  clusters_roads %>%
    ggplot(aes(
      x = PC1,
      y = PC2,
      color = cluster,
      size = viajes
    )) +
    geom_point(alpha = 0.75) +
    scale_color_brewer(palette = "Set2") +
    labs(
      x = "PC1",
      y = "PC2",
      color = "Cluster",
      size = "Viajes",
      title = "Clusters k-means sobre espacio PCA"
    )
} else {
  clusters_roads %>%
    ggplot(aes(
      x = PC1,
      y = cluster,
      color = cluster,
      size = viajes
    )) +
    geom_jitter(height = 0.12, alpha = 0.75) +
    scale_color_brewer(palette = "Set2") +
    labs(
      x = "PC1",
      y = "Cluster",
      color = "Cluster",
      size = "Viajes",
      title = "Clusters k-means sobre PC1"
    )
}

# ==============================================================================
# 5. PERFIL E INTERPRETACION DE LOS CLUSTERS
# ==============================================================================

# Aunque estas variables no participaron directamente en la formacion de los
# grupos, permiten describir sus caracteristicas sustantivas.
perfil_clusters <- clusters_roads %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    viajes_promedio = mean(viajes, na.rm = TRUE),
    viajeros_promedio = mean(viajeros, na.rm = TRUE),
    prop_weekend_promedio = mean(prop_weekend, na.rm = TRUE),
    prop_night_promedio = mean(prop_night, na.rm = TRUE),
    provincias = n_distinct(provincia),
    .groups = "drop"
  ) %>%
  arrange(desc(viajes_promedio))

write_csv(
  perfil_clusters,
  file.path(output_dir, "perfil_clusters_roads.csv")
)

perfil_clusters

# Convertimos el perfil a formato largo para comparar visualmente varios
# indicadores, aunque tengan escalas diferentes.
perfil_clusters %>%
  pivot_longer(
    cols = c(
      viajes_promedio,
      viajeros_promedio,
      prop_weekend_promedio,
      prop_night_promedio
    ),
    names_to = "variable",
    values_to = "valor"
  ) %>%
  ggplot(aes(x = cluster, y = valor, fill = cluster)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ variable, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    x = "Cluster",
    y = NULL,
    title = "Perfil promedio de clusters de movilidad"
  )

# Preguntas utiles para interpretar los resultados:
# - Los clusters separan por volumen total, patron temporal o ambos?
# - Que grupos tienen mayor movilidad relativa durante fines de semana?
# - Que grupos presentan mayor movilidad nocturna?
# - Los clusters se separan con claridad en PC1-PC2?
# - El k seleccionado produce perfiles utiles para politica publica?

# ==============================================================================
# 6. BASE ALTERNATIVA DEL INEC PARA EL TRABAJO APLICADO
# ==============================================================================

# Este bloque prepara indicadores socioeconomicos y de acceso a servicios para
# que el estudiante pueda repetir el flujo PCA + k-means con datos cantonales.
data_inec <- suppressMessages(
  read_excel("data/data_pobreza_INEC.xlsx")
)

# Renombramos variables extensas para facilitar recetas, graficos y perfiles.
inec_cluster <- data_inec %>%
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
    oficinas = `OficinasTasapobmay15años`
  )

write_csv(
  inec_cluster,
  file.path(output_dir, "inec_cluster_modulo4.csv")
)

glimpse(inec_cluster)

# Trabajo aplicado sugerido:
# 1. Crear datos_inec_cluster excluyendo canton. Tambien puede excluirse nbi si
#    se desea reservarlo exclusivamente para interpretar los grupos.
# 2. Crear una receta con imputacion, normalizacion y:
#       step_pca(all_numeric(), threshold = 0.70)
# 3. Repetir la evaluacion de k, el ajuste final y la visualizacion PCA.
# 4. Interpretar los clusters usando canton, nbi e indicadores originales.

# ==============================================================================
# 7. CONVERSION OPCIONAL DEL NOTEBOOK A HTML
# ==============================================================================

# El notebook original intentaba ejecutar nbconvert mediante un entorno conda.
# Hacemos el paso condicional para que la ausencia de conda no convierta en
# error una ejecucion correcta del analisis estadistico.
if (nzchar(Sys.which("conda"))) {
  system(
    "conda run -n rbase jupyter nbconvert --to html modulo4.ipynb"
  )
} else {
  message(
    "No se ejecuto nbconvert porque conda no esta disponible en PATH."
  )
}
