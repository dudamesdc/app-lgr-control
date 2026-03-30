import streamlit as st
import sympy as sp
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(page_title="LGR Passo a Passo", layout="wide")
st.title("Passo a Passo: Lugar Geométrico das Raízes")
st.markdown("Preencha os dados na barra lateral e clique em **Calcular** para ver a solução.")


st.sidebar.header("Entrada de Dados")
st.sidebar.markdown("---")

G_num_str = st.sidebar.text_input("Numerador G(s)", "s**2 + 0.2*s + 4.0")
G_den_str = st.sidebar.text_input("Denominador G(s)", "s**3 + 2*s**2 + 2*s + 1")
st.sidebar.markdown("---")
H_num_str = st.sidebar.text_input("Numerador H(s)", "1")
H_den_str = st.sidebar.text_input("Denominador H(s)", "s + 2")
st.sidebar.markdown("---")
s_teste_str = st.sidebar.text_input("Ponto de Teste", "3.2+2j")
st.sidebar.markdown("---")

calcular = st.sidebar.button("Calcular", use_container_width=True, type="primary")

if calcular:
    st.session_state["calculado"] = True
    st.session_state["G_num"] = G_num_str
    st.session_state["G_den"] = G_den_str
    st.session_state["H_num"] = H_num_str
    st.session_state["H_den"] = H_den_str
    st.session_state["s_teste_str"] = s_teste_str

if not st.session_state.get("calculado", False):
    st.stop()

G_num_val = st.session_state["G_num"]
G_den_val = st.session_state["G_den"]
H_num_val = st.session_state["H_num"]
H_den_val = st.session_state["H_den"]
s_teste_val = st.session_state["s_teste_str"]

try:
    # Preparação do sistema
    s = sp.symbols('s')
    K, w = sp.symbols('K w', real=True)

    # Construir polinômios a partir das strings
    G_num = sp.sympify(G_num_val)
    G_den = sp.sympify(G_den_val)
    H_num = sp.sympify(H_num_val)
    H_den = sp.sympify(H_den_val)

    # Ponto de teste
    s_teste = complex(s_teste_val) if s_teste_val.strip() != "" else None

    # Sistema
    G = G_num / G_den
    H = H_num / H_den
    P = sp.simplify(G * H)
    num_P, den_P = sp.fraction(P)

    
    st.markdown(f"**Malha Aberta $G(s)H(s)$:**")
    st.latex(sp.latex(P))


    abas = st.tabs([
        "Passos 1 a 3", 
        "Passos 4 a 6", 
        "Passo 7 (Assíntotas)", 
        "Passo 8 (Ramificação)", 
        "Passo 9 (Routh-Hurwitz)", 
        "Passo 10 (Ângulos)",
        "Passos 11 e 12 (Teste)", 
        "Gráfico Final"
    ])

    zeros_dict = sp.roots(num_P, s)
    polos_dict = sp.roots(den_P, s)
    zeros = [complex(z) for z, mult in zeros_dict.items() for _ in range(mult)]
    polos = [complex(p) for p, mult in polos_dict.items() for _ in range(mult)]
    np_polos, nz_zeros = len(polos), len(zeros)


    with abas[0]:
        
        st.subheader("Passo 1: Equação Característica")
        st.markdown(f"A equação característica da malha fechada é dada por $1 + G(s)H(s) = 0$:")
        st.markdown(f"$$1 + K \\cdot \\left( \\frac{{{sp.latex(num_P)}}}{{{sp.latex(den_P)}}} \\right) = 0$$")
        
        st.subheader("Passo 2: Polinômios Fatorados")
        st.markdown("Fatorando o numerador e o denominador, obtemos a representação evidenciando as raízes (polos e zeros):")
        num_P_factored = sp.factor(num_P, s)
        den_P_factored = sp.factor(den_P, s)
        st.markdown(f"$$1 + K \\cdot \\left( \\frac{{{sp.latex(num_P_factored)}}}{{{sp.latex(den_P_factored)}}} \\right) = 0$$")

        st.subheader("Passo 3: Polos e Zeros")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Zeros (Chegadas - O):**")
            for i, z in enumerate(zeros):
                st.code(f"z{i+1} = {np.round(z.real, 2)} + {np.round(z.imag, 2)}j", language='text')
            if not zeros: st.write("Não há zeros.")
        with col2:
            st.write("**Polos (Partidas - X):**")
            for i, p in enumerate(polos):
                st.code(f"p{i+1} = {np.round(p.real, 2)} + {np.round(p.imag, 2)}j", language='text')

  
    with abas[1]:
        st.markdown(f"- **Número de Lugares Separados (LS = Np): {max(np_polos, nz_zeros)}**")
        
        elementos_reais = sorted([val.real for val in polos + zeros if abs(val.imag) < 1e-5], reverse=True)
        if not elementos_reais:
            st.info("Não há polos ou zeros puramente reais.")
        else:
            st.write("**Intervalos no eixo real (Regra do número ímpar à direita):**")
            for i in range(len(elementos_reais)):
                lim_dir = elementos_reais[i]
                lim_esq = elementos_reais[i+1] if i+1 < len(elementos_reais) else "-∞"
                qtd = i + 1
                status = "EXISTE LGR" if qtd % 2 != 0 else "NÃO existe LGR"
                st.code(f"Intervalo [ {lim_dir:.2f} , {lim_esq} ] -> {qtd} elementos à direita ({status})", language='text')


    with abas[2]:
        if np_polos > nz_zeros:
            num_assintotas = np_polos - nz_zeros
            soma_polos = sum(polos)
            soma_zeros = sum(zeros)
            centroide = (soma_polos - soma_zeros) / num_assintotas
            
            st.markdown(f"- **Quantidade (Np-Nz):** ${np_polos} - {nz_zeros} = {num_assintotas}$ assíntotas")
            st.write("**Cálculo do Centroide (σA):**")
            st.code(f"σA = ( Σpolos - Σzeros ) / {num_assintotas}\nσA = ( {soma_polos.real:.2f} - {soma_zeros.real:.2f} ) / {num_assintotas} = {centroide.real:.3f}", language='text')
            
            st.write("**Ângulos (φA):**")
            for q in range(num_assintotas):
                ang = ((2*q + 1)*180) / num_assintotas
                st.code(f"Para q={q}: φA = (2({q})+1)*180 / {num_assintotas} = {ang}°", language='text')
        else:
            st.info("Não há assíntotas (zeros >= polos).")


    with abas[3]:
        N_s = num_P
        D_s = den_P

        dN_ds = sp.diff(N_s, s)
        dD_ds = sp.diff(D_s, s)
        poly_ramificacao = sp.expand(dD_ds * N_s - D_s * dN_ds)
        st.latex(sp.latex(poly_ramificacao) + " = 0")
        pontos_ram = sp.solve(poly_ramificacao, s)
        
        st.write("**Soluções calculadas (Candidatos a Ramos):**")
        if not pontos_ram:
            st.info("O sistema não possui raízes para a derivada, portanto não possui pontos de ramificação.")
        else:
            for i, p in enumerate(pontos_ram):

                val = complex(p.evalf())
   
                if abs(val.imag) < 1e-5:
                    st.code(f"s_{i+1} = {val.real:.4f}", language='text')
                else:
                    st.code(f"s_{i+1} = {val.real:.4f} + {val.imag:.4f}j", language='text')

    with abas[4]:
        st.header("Passo 9: Cruzamento com Eixo Imaginário (Routh-Hurwitz)")
        poly_carac = sp.collect(sp.expand(den_P + K * num_P), s)
        st.write("**Equação Característica Expandida:**")
        st.latex(sp.latex(poly_carac) + " = 0")

        coeffs = sp.Poly(poly_carac, s).all_coeffs()
        grau = len(coeffs) - 1
        row0 = [coeffs[i] for i in range(0, len(coeffs), 2)]
        row1 = [coeffs[i] for i in range(1, len(coeffs), 2)]
        max_len = max(len(row0), len(row1))
        row0 += [0] * (max_len - len(row0))
        row1 += [0] * (max_len - len(row1))
        rh_table = [row0, row1]

        for i in range(2, grau + 1):
            new_row = []
            for j in range(max_len - 1):
                c1, c2 = rh_table[i-1][0], rh_table[i-2][j+1]
                c3, c4 = rh_table[i-2][0], rh_table[i-1][j+1]
                elem = sp.simplify((c1*c2 - c3*c4) / c1)
                new_row.append(elem)
            new_row.append(0)
            rh_table.append(new_row)

        tabela_display = []
        for idx, row in enumerate(rh_table):
            linha_str = [str(sp.simplify(val)) if val != 0 else "0" for val in row]
            tabela_display.append([f"s^{grau - idx}"] + linha_str)
        
        df_rh = pd.DataFrame(tabela_display)
        st.table(df_rh)

        eq_s1 = rh_table[grau - 1][0]
        st.write("**Cálculo do Ganho Crítico:** Igualar linha $s^1$ a zero:")
        st.latex(sp.latex(eq_s1) + " = 0")
        try:
            k_crit_vals = sp.solve(eq_s1, K)
            k_validos = [float(k.evalf()) for k in k_crit_vals if k.evalf() > 0]
            if k_validos:
                for k_val in k_validos:
                    st.code(f"K crítico = {k_val:.4f}", language='text')
                    linha_s2_A = rh_table[grau - 2][0].subs(K, k_val)
                    linha_s2_B = rh_table[grau - 2][1].subs(K, k_val)
                    w2 = float(linha_s2_B / linha_s2_A)
                    st.write("**Frequência de Oscilação:** (Substituindo K na linha $s^2$)")
                    st.code(f"{linha_s2_A:.4f}s^2 + {linha_s2_B:.4f} = 0  =>  s = ± {np.sqrt(w2):.4f}j", language='text')
            else:
                st.info("Não há ganho crítico positivo.")
        except Exception as e:
            st.error("Não foi possível resolver algebricamente.")


    with abas[5]:
        st.header("Passo 10: Ângulos de Partida e Chegada")
        complex_polos = [p for p in polos if abs(p.imag) > 1e-5]
        complex_zeros = [z for z in zeros if abs(z.imag) > 1e-5]

        if not complex_polos and not complex_zeros:
            st.info("Não há polos ou zeros complexos. LGR sai/chega pelo eixo real.")
        else:
            if complex_polos:
                st.write("**1) Ângulo de Partida (Polos Complexos):**")
                for pk in complex_polos:
                    if pk.imag > 0:
                        ang_p = [np.degrees(np.angle(pk - pi)) for pi in polos if not np.isclose(pk, pi)]
                        ang_z = [np.degrees(np.angle(pk - zi)) for zi in zeros]
                        ang_partida = (180 - sum(ang_p) + sum(ang_z)) % 360
                        st.code(f"Para polo {np.round(pk,2)}:\nθ_partida = 180° - Σθ + Σφ = {ang_partida:.2f}°", language='text')

            if complex_zeros:
                st.write("**2) Ângulo de Chegada (Zeros Complexos):**")
                for zk in complex_zeros:
                    if zk.imag > 0:
                        ang_z = [np.degrees(np.angle(zk - zi)) for zi in zeros if not np.isclose(zk, zi)]
                        ang_p = [np.degrees(np.angle(zk - pi)) for pi in polos]
                        ang_chegada = (180 - sum(ang_z) + sum(ang_p)) % 360
                        st.code(f"Para zero {np.round(zk,2)}:\nφ_chegada = 180° - Σφ + Σθ = {ang_chegada:.2f}°", language='text')


    with abas[6]:
        if s_teste:
            st.header(f"Passos 11 e 12: Análise do Ponto de Teste $s = {s_teste}$")
            soma_ang_zeros, soma_ang_polos = 0, 0
            prod_dist_zeros, prod_dist_polos = 1.0, 1.0

            st.subheader("Passo 11: Critério do Ângulo")
            st.markdown("Para que um ponto pertença ao Lugar Geométrico das Raízes (LGR), a soma dos ângulos dos vetores traçados dos zeros até o ponto menos a soma dos ângulos dos vetores traçados dos polos até o ponto deve ser um múltiplo ímpar de 180°")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Ângulos a partir dos Zeros (φ):**")
                if not zeros:
                    st.write("Não há zeros.")
                for i, z in enumerate(zeros):
                    vetor = s_teste - z
                    ang = np.degrees(np.angle(vetor))
                    soma_ang_zeros += ang
                    st.code(f"s - z{i+1} = {np.round(vetor,2)}\nφ{i+1} = {ang:.2f}°", language='text')
            with c2:
                st.write("**Ângulos a partir dos Polos (θ):**")
                if not polos:
                    st.write("Não há polos.")
                for j, p in enumerate(polos):
                    vetor = s_teste - p
                    ang = np.degrees(np.angle(vetor))
                    soma_ang_polos += ang
                    st.code(f"s - p{j+1} = {np.round(vetor,2)}\nθ{j+1} = {ang:.2f}°", language='text')
            
            ang_total_bruto = soma_ang_zeros - soma_ang_polos
            ang_total = (soma_ang_zeros - soma_ang_polos) % 360
            
            if ang_total > 180:
                ang_total_norm = ang_total - 360
            else:
                ang_total_norm = ang_total

            st.markdown("**Cálculo da Condição Angular:**")
            st.markdown(f"$\\Sigma\\phi - \\Sigma\\theta = {soma_ang_zeros:.2f}^\\circ - {soma_ang_polos:.2f}^\\circ = {ang_total_bruto:.2f}^\\circ$")
            st.markdown(f"Ângulo normalizado: **{ang_total_norm:.2f}°**")
            
            if abs(abs(ang_total_norm) - 180) < 5.0 or abs(ang_total_norm) < 5.0:
                st.success(f"O ângulo resultante ({ang_total_norm:.2f}°) satisfaz a condição de ângulo. O ponto **PERTENCE** ao LGR!")
                
                st.subheader("Passo 12: Critério do Módulo")
                st.markdown("Como o ponto pertence ao LGR, podemos determinar o valor do ganho $K$ usando o critério do módulo:")
                st.markdown(r"$$K = \frac{\prod |s - p_i|}{\prod |s - z_i|}$$")
                
                c3, c4 = st.columns(2)
                with c3:
                    st.write("**Distâncias dos Zeros:**")
                    if not zeros:
                         st.write("Sem zeros (produto = 1).")
                    for i, z in enumerate(zeros):
                         dist = abs(s_teste - z)
                         prod_dist_zeros *= dist
                         st.code(f"|s - z{i+1}| = {dist:.4f}", language='text')
                with c4:
                    st.write("**Distâncias dos Polos:**")
                    if not polos:
                         st.write("Sem polos (produto = 1).")
                    for j, p in enumerate(polos):
                         dist = abs(s_teste - p)
                         prod_dist_polos *= dist
                         st.code(f"|s - p{j+1}| = {dist:.4f}", language='text')
                         
                K_calc = prod_dist_polos / prod_dist_zeros
                st.markdown("**Cálculo do Ganho $K$:**")
                st.code(f"K = {prod_dist_polos:.4f} / {prod_dist_zeros:.4f} = {K_calc:.4f}", language='text')

            else:
                st.error(f"O ângulo resultante ({ang_total_norm:.2f}°) não é um múltiplo ímpar de 180°. O ponto **NÃO PERTENCE** ao LGR.")
                st.info("O Passo 12 (Critério do Módulo) não é calculado, pois o ponto não atende ao Critério do Ângulo.")
        else:
            st.warning("Nenhum ponto de teste inserido na barra lateral.")


    with abas[7]:
        st.header("Esboço Computacional do LGR")
        
        num_P_poly = [float(c) for c in sp.Poly(num_P, s).all_coeffs()]
        den_P_poly = [float(c) for c in sp.Poly(den_P, s).all_coeffs()]
        sys = ct.TransferFunction(num_P_poly, den_P_poly)

        fig = plt.figure(figsize=(8, 6))
        ct.root_locus(sys, grid=True)
        
        # Resolve o alerta do matplotlib sobre conflict de aspect ratio "Ignoring fixed x limits..."
        fig.gca().set_adjustable("datalim")

        if s_teste is not None:
            plt.plot(s_teste.real, s_teste.imag, 'r*', markersize=10, label=f'Teste: {s_teste}')

        plt.title("Lugar Geométrico das Raízes (LGR)")
        plt.xlabel("Eixo Real")
        plt.ylabel("Eixo Imaginário")
        plt.legend()
        
        st.pyplot(fig)
        
        plt.close(fig)

except Exception as e:
    st.error(f"Ocorreu um erro no processamento das equações. Verifique a sintaxe. Erro: {e}")