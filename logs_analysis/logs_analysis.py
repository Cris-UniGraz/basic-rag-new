import json
import os
import csv
from pathlib import Path

# Uso: python logs_analysis.py

def create_directories():
    """Crea el directorio output si no existe"""
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def format_decimal(value, decimals=3):
    """Formatea un número decimal con comas como separador decimal"""
    return f"{value:.{decimals}f}".replace('.', ',')

def format_percentage(part, total, decimals=2):
    """Calcula y formatea un porcentaje con comas como separador decimal"""
    if total == 0:
        return "0,00%"
    percentage = (part / total) * 100
    return f"{percentage:.{decimals}f}%".replace('.', ',')

def process_jsonl_file(file_path):
    """Procesa el archivo JSONL y extrae los datos de performance"""
    performance_data = []
    query_count = 1
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                
                # Verificar si es una línea de "Async pipeline performance summary"
                if (data.get('data', {}).get('message') == 'Async pipeline performance summary'):
                    extra = data['data']['extra']
                    
                    # Extraer datos principales
                    query = extra.get('query', '')
                    query_was_answered = extra.get('query_was_answered', 'False')
                    total_time = extra.get('total_time', 0)
                    phase_breakdown = extra.get('phase_breakdown', {})
                    
                    # Determinar si la respuesta fue generada
                    # print(f"Procesando query: {query} (Answered: {query_was_answered})")
                    answer_symbol = "✓" if query_was_answered else "✗"
                    
                    # Extraer tiempos de cada fase
                    cache_optimization = phase_breakdown.get('cache_optimization', 0)
                    query_generation = phase_breakdown.get('query_generation', 0)
                    retrieval = phase_breakdown.get('retrieval', 0)
                    processing_reranking = phase_breakdown.get('processing_reranking', 0)
                    response_preparation = phase_breakdown.get('response_preparation', 0)
                    llm_generation = phase_breakdown.get('llm_generation', 0)
                    
                    # Crear fila de datos
                    row = [
                        f"{query_count})",  # Número de query
                        query,              # Query/Anfrage
                        answer_symbol,  # Respuesta generada?
                        format_decimal(total_time),  # Total time
                        format_decimal(cache_optimization),  # Phase 1 time
                        format_percentage(cache_optimization, total_time),  # Phase 1 %
                        format_decimal(query_generation),  # Phase 2 time
                        format_percentage(query_generation, total_time),  # Phase 2 %
                        format_decimal(retrieval),  # Phase 3 time
                        format_percentage(retrieval, total_time),  # Phase 3 %
                        format_decimal(processing_reranking),  # Phase 4 time
                        format_percentage(processing_reranking, total_time),  # Phase 4 %
                        format_decimal(response_preparation, 5),  # Phase 5 time
                        format_percentage(response_preparation, total_time),  # Phase 5 %
                        format_decimal(llm_generation),  # Phase 6 time
                        format_percentage(llm_generation, total_time)  # Phase 6 %
                    ]
                    
                    performance_data.append(row)
                    query_count += 1
                    
            except json.JSONDecodeError:
                # Ignorar líneas que no sean JSON válido
                continue
            except Exception as e:
                print(f"Error procesando línea: {e}")
                continue
    
    return performance_data

def create_csv_output(performance_data, output_path):
    """Crea el archivo CSV con el formato requerido"""
    
    # Encabezados según el formato del archivo original
    headers = [
        ["Async pipeline performance summary", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Anfrage", "", "Answer generated?", "Total Time [s]", "Phase 1:\nCache Optimization", "", "Phase 2:\nQuery Pre-processing", "", "Phase 3:\nRetrieval", "", "Phase 4:\nReranking", "", "Phase 5:\nResponse Preparation", "", "Phase 6:\nLLM Generation", ""]
    ]
    
    # Cambiar encoding a 'utf-8-sig' para incluir BOM
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        
        # Escribir encabezados
        for header in headers:
            writer.writerow(header)
        
        # Escribir datos de performance
        for row in performance_data:
            writer.writerow(row)
        
        # Añadir filas vacías al final como en el archivo original
        for _ in range(3):
            writer.writerow(["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

def main():
    """Función principal"""
    try:
        # Crear directorios necesarios
        output_dir = create_directories()
        
        # Rutas de archivos
        input_file = Path("logs/async_logs.jsonl") # "logs/async_logs_gpt-4.1-mini.jsonl" # "logs/async_logs_gpt-4.1-nano.jsonl"
        output_file = output_dir / "Performance_rag_project_analysis.csv"
        
        # Verificar que el archivo de entrada existe
        if not input_file.exists():
            print(f"Error: El archivo {input_file} no existe.")
            return
        
        print("Procesando archivo JSONL...")
        
        # Procesar el archivo JSONL
        performance_data = process_jsonl_file(input_file)
        
        if not performance_data:
            print("No se encontraron datos de performance en el archivo JSONL.")
            return
        
        print(f"Se encontraron {len(performance_data)} registros de performance.")
        
        # Crear archivo CSV
        create_csv_output(performance_data, output_file)
        
        print(f"Archivo CSV creado exitosamente: {output_file}")
        print(f"Total de queries procesadas: {len(performance_data)}")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
