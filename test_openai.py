import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError

st.title("Prueba de conexión con OpenAI")

if st.button("Probar conexión"):
    try:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
        model = str(st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"))

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Respondé solamente: Conexión correcta."}
            ],
            max_tokens=20,
        )

        st.success("La conexión funcionó.")
        st.write(response.choices[0].message.content)

    except KeyError:
        st.error("No se encontró OPENAI_API_KEY en los Secrets de esta app.")
    except AuthenticationError:
        st.error("OpenAI rechazó la clave guardada en Secrets (error 401).")
    except RateLimitError:
        st.error("La solicitud fue limitada. Revisá el acceso disponible para la API.")
    except APIConnectionError:
        st.error("No se pudo conectar con OpenAI. Probá nuevamente más tarde.")
    except Exception as exc:
        st.error(f"Falló la prueba: {type(exc).__name__}")
