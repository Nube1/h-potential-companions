options(stringsAsFactors = FALSE)

required_packages <- c("lavaan", "survival", "mice")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install required R packages: ", paste(missing_packages, collapse = ", "))
}

data_path <- Sys.getenv("HPOTENTIAL_DATA", unset = "data/nhats_analysis.csv")
dictionary_path <- "analysis/data_dictionary.csv"
if (!file.exists(data_path)) {
  stop("NHATS analysis extract not found at '", data_path,
       "'. Set HPOTENTIAL_DATA to the approved de-identified extract.")
}
if (!file.exists(dictionary_path)) stop("Missing analysis/data_dictionary.csv")

dictionary <- read.csv(dictionary_path, check.names = FALSE)
dat <- read.csv(data_path, check.names = FALSE)
required_columns <- dictionary$analysis_name[dictionary$required == "yes"]
missing_columns <- setdiff(required_columns, names(dat))
if (length(missing_columns)) {
  stop("Required columns missing: ", paste(missing_columns, collapse = ", "))
}

dir.create("outputs", showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/session-info.txt")
message("Preflight passed: ", nrow(dat), " rows; ", ncol(dat), " columns.")
