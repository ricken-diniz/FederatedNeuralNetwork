import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from modelNet import Net
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def verify_x_is_y(x, y, max_samples=50):
    
    model = Net()
    model.load_state_dict(torch.load('federated_model.pt'))
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True, **{})

    print(f"=== TESTE DE POISONING: Dígitos '{x}' ===")
    print(f"Verificando se o modelo classifica '{x}' como '{y}'...\n")
    
    # Listas para armazenar os dados
    results_data = []
    count = len(test_loader)
    correct_x = 0  # Quantos xs foram classificados corretamente como x
    poisoned_x = 0  # Quantos xs foram classificados como y (poisoning)
    other_wrong = 0  # Quantos xs foram classificados como outros números
    
    with torch.no_grad():
        for data, target in test_loader:
            if target[0].item() == x:
                data, target = data.to('cpu'), target.to('cpu')
                
                # Faz a predição
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                predicted = output.argmax(dim=1, keepdim=True)
                predicted_class = predicted[0].item()
                confidence = probabilities[0][predicted_class].item()
                
                # Armazena dados para gráficos
                results_data.append({
                    'amostra': count + 1,
                    'real': x,
                    'predito': predicted_class,
                    'confianca': confidence,
                    'status': 'Correto' if predicted_class == x else 
                             'Poisoning' if predicted_class == y else 'Erro'
                })
                
                # Mostra resultado
                print(f"Amostra {count + 1}:")
                print(f"  Real: {x}")
                print(f"  Predito: {predicted_class}")
                print(f"  Confiança: {confidence:.4f}")
                
                # Contabiliza resultado
                if predicted_class == x:
                    correct_x += 1
                    print("  Status: ✓ CORRETO")
                elif predicted_class == y:
                    poisoned_x += 1
                    print("  Status: ⚠️ POISONING DETECTADO!")
                else:
                    other_wrong += 1
                    print(f"  Status: ✗ ERRO (classificado como {predicted_class})")
                
                print("-" * 40)
                
    
    # Resumo dos resultados
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(f"Total de amostras testadas: {count}")
    print(f"Classificações corretas ({x} → {x}): {correct_x} ({correct_x/count*100:.1f}%)")
    print(f"Poisoning detectado ({x} → {y}): {poisoned_x} ({poisoned_x/count*100:.1f}%)")
    print(f"Outros erros: {other_wrong} ({other_wrong/count*100:.1f}%)")
    
    if poisoned_x > 0:
        print(f"\n🚨 ALERTA: Modelo foi comprometido por poisoning!")
        print(f"Taxa de sucesso do ataque: {poisoned_x/count*100:.1f}%")
    else:
        print(f"\n✅ Modelo não apresenta sinais de poisoning nos testes.")
    
    # Criar gráficos
    if results_data:
        create_graphs(results_data, x, y)
    
    return results_data

def create_graphs(results_data, target_digit, poison_digit):
    """Cria gráficos das predições do modelo"""
    df = pd.DataFrame(results_data)
    
    # Configurar subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análise de Poisoning: Dígito {target_digit} → {poison_digit}', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Predições por amostra
    colors = {'Correto': 'green', 'Poisoning': 'red', 'Erro': 'orange'}
    for status, color in colors.items():
        mask = df['status'] == status
        if mask.any():
            ax1.scatter(df[mask]['amostra'], df[mask]['predito'], 
                       c=color, label=status, alpha=0.7, s=60)
    
    ax1.axhline(y=target_digit, color='blue', linestyle='--', alpha=0.5, label=f'Valor Real ({target_digit})')
    ax1.axhline(y=poison_digit, color='red', linestyle='--', alpha=0.5, label=f'Poisoning ({poison_digit})')
    ax1.set_xlabel('Amostra')
    ax1.set_ylabel('Predição')
    ax1.set_title('Predições por Amostra')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 9.5)
    
    # Gráfico 2: Distribuição das predições
    prediction_counts = df['predito'].value_counts().sort_index()
    bars = ax2.bar(prediction_counts.index, prediction_counts.values, 
                   color=['red' if x == poison_digit else 'green' if x == target_digit else 'lightblue' 
                          for x in prediction_counts.index])
    
    # Destacar as barras importantes
    for i, bar in enumerate(bars):
        digit = prediction_counts.index[i]
        if digit == target_digit:
            bar.set_color('green')
            bar.set_alpha(0.8)
        elif digit == poison_digit:
            bar.set_color('red') 
            bar.set_alpha(0.8)
    
    ax2.set_xlabel('Dígito Predito')
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Distribuição das Predições')
    ax2.set_xticks(range(10))
    
    # Adicionar valores nas barras
    for i, v in enumerate(prediction_counts.values):
        ax2.text(prediction_counts.index[i], v + 0.1, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Confiança das predições
    ax3.scatter(df['amostra'], df['confianca'], 
               c=[colors.get(status, 'gray') for status in df['status']], alpha=0.7)
    ax3.set_xlabel('Amostra')
    ax3.set_ylabel('Confiança')
    ax3.set_title('Confiança das Predições')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Gráfico 4: Pizza dos resultados
    status_counts = df['status'].value_counts()
    colors_pie = [colors[status] for status in status_counts.index]
    wedges, texts, autotexts = ax4.pie(status_counts.values, labels=status_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax4.set_title('Distribuição dos Resultados')
    
    # Melhorar a aparência do gráfico de pizza
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Salvar o gráfico
    filename = f'poisoning_analysis_{target_digit}_to_{poison_digit}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráficos salvos em: {filename}")
    
    plt.show()
    
    # Estatísticas detalhadas
    print("\n=== ESTATÍSTICAS DETALHADAS ===")
    print("Distribuição das predições:")
    for digit, count in prediction_counts.items():
        percentage = (count / len(df)) * 100
        status = "🎯 CORRETO" if digit == target_digit else "⚠️ POISONING" if digit == poison_digit else "❌ ERRO"
        print(f"  Dígito {digit}: {count} amostras ({percentage:.1f}%) {status}")
    
    print(f"\nConfiança média:")
    for status in df['status'].unique():
        avg_conf = df[df['status'] == status]['confianca'].mean()
        print(f"  {status}: {avg_conf:.3f}")
    
    return df

def main():
    """Função principal para testar o modelo"""
    # Testa se dígitos 7 estão sendo classificados como 1 (poisoning)
    print("Iniciando análise de poisoning...")
    results = verify_x_is_y(x=7, y=1, max_samples=50)
    
    # Opcionalmente, teste outros dígitos para comparação
    # results_normal = verify_x_is_y(x=5, y=1, max_samples=30)  # Teste de controle

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    verify_x_is_y(7,1)
