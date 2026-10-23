# 출력 확인표

필수 결과: outputs/split_manifest.csv, feature_names.json, validation_predictions.csv, validation_features.csv,
validation_metrics.json, preprocessing_state.json, run_metadata.json, environment.json, protocol_log.json,
split_order.json, new_predictions.csv, pipeline.joblib, restore_result.json, artifact_manifest.json, requirements-lock.txt.

실행 후 생성되는 파일입니다. 배포 ZIP에는 비교용 텍스트/CSV/JSON만 있으며 학습된 joblib은 들어 있지 않습니다.
`pipeline.joblib`은 본인이 직접 실행해 생성한 파일만 읽으세요. `artifact_manifest.json` 해시는 우발적 변경 탐지용입니다.
README·REPORT.md·TEST_LOG.md는 분석·재현의 근거를 작성할 소스 문서이며 자동 출력으로 대신하지 않습니다.
