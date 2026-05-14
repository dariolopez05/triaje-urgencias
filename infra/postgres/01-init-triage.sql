CREATE TABLE IF NOT EXISTS Entrevista (
    GUID_Entrevista              VARCHAR(255) PRIMARY KEY,
    ID_CASO                      VARCHAR(50),
    Origen                       VARCHAR(20)  CHECK (Origen IN ('Dataset','Simulacion','MVP')),
    URL_Audio_Original           VARCHAR(255),
    URL_Texto_Original           VARCHAR(255),
    URL_Dataset_Generado         VARCHAR(255),
    URL_Modelo_Entrenado         VARCHAR(255),
    Inicio_Solicitud             TIMESTAMP,
    Fin_Solicitud                TIMESTAMP,
    Inicio_Transcripcion         TIMESTAMP,
    Fin_Transcripcion            TIMESTAMP,
    Inicio_Preprocesamiento      TIMESTAMP,
    Fin_Preprocesamiento         TIMESTAMP,
    Inicio_Extraccion_Entidades  TIMESTAMP,
    Fin_Extraccion_Entidades     TIMESTAMP,
    Inicio_Normalizacion         TIMESTAMP,
    Fin_Normalizacion            TIMESTAMP,
    Inicio_Etiquetado            TIMESTAMP,
    Fin_Etiquetado               TIMESTAMP,
    Inicio_Score                 TIMESTAMP,
    Fin_Score                    TIMESTAMP,
    Inicio_Entrenamiento         TIMESTAMP,
    Fin_Entrenamiento            TIMESTAMP,
    Motor_Workflow               VARCHAR(50),
    Workflow_Id                  VARCHAR(255),
    Estado                       VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_entrevista_estado ON Entrevista (Estado);

CREATE TABLE IF NOT EXISTS Texto_Procesado (
    guid                          VARCHAR(255) PRIMARY KEY
                                  REFERENCES Entrevista(GUID_Entrevista) ON DELETE CASCADE,
    texto_original_en             TEXT,
    resumen_es                    TEXT,
    texto_preprocesado            TEXT,
    entidades_extraidas_es        JSONB,
    entidades_normalizadas_es     JSONB,
    triage_real                   VARCHAR(5)
                                  CHECK (triage_real IN ('C1','C2','C3','C4','C5')),
    score_ansiedad                NUMERIC(3,2)
                                  CHECK (score_ansiedad BETWEEN 0 AND 1),
    justificacion_llm             TEXT
);

CREATE TABLE IF NOT EXISTS Prediccion (
    guid                VARCHAR(255) PRIMARY KEY
                        REFERENCES Entrevista(GUID_Entrevista) ON DELETE CASCADE,
    prediccion_ia       VARCHAR(5)
                        CHECK (prediccion_ia IN ('C1','C2','C3','C4','C5')),
    score_ansiedad_ia   NUMERIC(3,2) CHECK (score_ansiedad_ia BETWEEN 0 AND 1),
    validacion          VARCHAR(20)
                        CHECK (validacion IN ('Acierto','Under-triage','Over-triage','Pendiente')),
    motivo_fallo        TEXT,
    accion_correctiva   TEXT,
    fecha               TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS Task_Log (
    id                  SERIAL PRIMARY KEY,
    guid                VARCHAR(255),
    service_name        VARCHAR(100) NOT NULL,
    timestamp_inicio    TIMESTAMP NOT NULL,
    timestamp_fin       TIMESTAMP,
    status              VARCHAR(20)
                        CHECK (status IN ('OK','ERROR','RETRY','TIMEOUT')),
    payload_resultado   JSONB,
    error_msg           TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_log_guid    ON Task_Log (guid);
CREATE INDEX IF NOT EXISTS idx_task_log_service ON Task_Log (service_name);
CREATE INDEX IF NOT EXISTS idx_task_log_status  ON Task_Log (status);

CREATE OR REPLACE VIEW v_resultado_completo AS
SELECT
    e.GUID_Entrevista,
    e.ID_CASO,
    e.Origen,
    e.Estado,
    t.resumen_es,
    t.entidades_normalizadas_es,
    t.triage_real,
    t.score_ansiedad,
    t.justificacion_llm,
    p.prediccion_ia,
    p.score_ansiedad_ia,
    p.validacion,
    p.motivo_fallo
FROM Entrevista e
LEFT JOIN Texto_Procesado t ON e.GUID_Entrevista = t.guid
LEFT JOIN Prediccion       p ON e.GUID_Entrevista = p.guid;
