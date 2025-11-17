# generate_synthetic_table

LangGraph 기반의 Agentic 플로우를 이용해 한국어 표 이미지를 합성 데이터로 변환합니다. 입력 이미지를 HTML 표로 변환한 뒤, 내용을 분석하여 동일한 구조를 가진 라이선스 프리 합성 데이터를 생성합니다.

## 구성

플로우는 다음 4단계로 구성되어 있습니다.

1. **Image2HTML** – 표 이미지를 HTML `<table>` 구조로 복원합니다.
2. **Parse Contents** – 표의 열/행 정보를 요약하고 패턴을 추출합니다.
3. **Generate Synthetic Dataset** – 동일한 구조를 유지하면서 합성 데이터를 채운 HTML 표를 생성합니다.
4. **Self-Reflection** – 생성된 표가 라이선스/개인정보 이슈가 없는지 점검합니다.

## 설치

프로젝트 루트에 `.env` 파일을 만들고 OpenAI 키를 설정합니다.

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

의존성은 `pyproject.toml`을 통해 관리되므로 원하는 패키지 매니저로 설치합니다. `uv`를 사용하는 경우:

```bash
uv sync
```

또는 일반 `pip`를 사용할 경우:

```bash
pip install .
```

## 사용법

명령행 인터페이스를 이용해 플로우를 실행할 수 있습니다.

```bash
python main.py /path/to/table-image.png --save-json result.json
```

- `--model`: 사용할 OpenAI 모델 이름 (기본값: `gpt-4.1-mini`)
- `--temperature`: 모델 온도 (기본값: `0.2`)
- `--save-json`: 최종 상태를 JSON 파일로 저장합니다.

`main.py`는 내부의 `src.runner` 모듈을 사용해 동일한 로직을 실행하므로, 다른 스크립트나 노트북에서도 `run_flow_for_image` 함수를 가져다가 간단히 재사용할 수 있습니다.

실행 결과에는 HTML 표, 내용 요약, 합성 표, 자기 점검 결과가 포함됩니다.
