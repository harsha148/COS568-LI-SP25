import pandas as pd


def result_analysis():
    tasks = ['fb', 'osmc', 'books']
    indexs = ['HybridPGMLIPP', 'DynamicPGM', 'LIPP']
    # Create dictionaries to store throughput data for each index
    # lookuponly_throughput = {}
    # insertlookup_throughput = {}
    insertlookup_mix1_throughput = {}
    insertlookup_mix2_throughput = {}
    
    for index in indexs:
        # lookuponly_throughput[index] = {}
        # insertlookup_throughput[index] = {"lookup": {}, "insert": {}}
        insertlookup_mix1_throughput[index] = {}
        insertlookup_mix2_throughput[index] = {}
    
    for task in tasks:
        full_task_name = f"{task}_100M_public_uint64"
        # lookup_only_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.000000i_results_table.csv")
        # insert_lookup_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.500000i_0m_results_table.csv")
        insert_lookup_mix_1_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.100000i_0m_mix_results_table.csv")
        insert_lookup_mix_2_results = pd.read_csv(f"results/{full_task_name}_ops_2M_0.000000rq_0.500000nl_0.900000i_0m_mix_results_table.csv")
        
        for index in indexs:
            # # find the row where lookup_only_result['index_name'] == index
            # try:
            #     lookup_only_result = lookup_only_results[lookup_only_results['index_name'] == index]
            #     # compute average throughput across lookup_only_result['throughput1'], lookup_only_result['throughput2'], lookup_only_result['throughput3'], then select the one with the highest throughput
            #     lookuponly_throughput[index][task] = lookup_only_result[['lookup_throughput_mops1', 'lookup_throughput_mops2', 'lookup_throughput_mops3']].mean(axis=1).max()
            # except:
            #     pass
            #
            # # find the row where insert_lookup_result['index_name'] == index
            # try:
            #     insert_lookup_result = insert_lookup_results[insert_lookup_results['index_name'] == index]
            #     # compute average throughput across insert_lookup_result['throughput1'], insert_lookup_result['throughput2'], insert_lookup_result['throughput3'], then select the one with the highest throughput
            #     insertlookup_throughput[index]['lookup'][task] = insert_lookup_result[['lookup_throughput_mops1', 'lookup_throughput_mops2', 'lookup_throughput_mops3']].mean(axis=1).max()
            #     insertlookup_throughput[index]['insert'][task] = insert_lookup_result[['insert_throughput_mops1', 'insert_throughput_mops2', 'insert_throughput_mops3']].mean(axis=1).max()
            # except:
            #     pass
            
                
            # find the row where insert_lookup_mix_1_result['index_name'] == index
            try:
                insert_lookup_mix_1_result = insert_lookup_mix_1_results[insert_lookup_mix_1_results['index_name'] == index]
                # compute average throughput across insert_lookup_mix_1_result['throughput1'], insert_lookup_mix_1_result['throughput2'], insert_lookup_mix_1_result['throughput3'], then select the one with the highest throughput
                insertlookup_mix1_throughput[index][task] = insert_lookup_mix_1_result[['mixed_throughput_mops1', 'mixed_throughput_mops2', 'mixed_throughput_mops3']].mean(axis=1).max()
            except:
                pass
            
            
            # find the row where insert_lookup_mix_2_result['index_name'] == index
            try:
                insert_lookup_mix_2_result = insert_lookup_mix_2_results[insert_lookup_mix_2_results['index_name'] == index]
                # compute average throughput across insert_lookup_mix_2_result['throughput1'], insert_lookup_mix_2_result['throughput2'], insert_lookup_mix_2_result['throughput3'], then select the one with the highest throughput
                insertlookup_mix2_throughput[index][task] = insert_lookup_mix_2_result[['mixed_throughput_mops1', 'mixed_throughput_mops2', 'mixed_throughput_mops3']].mean(axis=1).max()
            except:
                pass
    # plot the figure of throughput, x axis is the index, y axis is the throughput
    # the figure should contain 4 subplots, each subplot corresponds to a workload, including lookup_only, insert_lookup, insert_lookup_mix1, insert_lookup_mix2
    # each subplot should contain 3 bars, each bar corresponds to a dataset (fb, osmc, books) if the throughput is not empty
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    # Flatten axs for easier indexing
    axs = axs.flatten()
    
    # Define common plot parameters
    bar_width = 0.2
    index = range(len(indexs))
    colors = ['blue', 'green', 'red', 'orange']
    
    # # 1. Plot lookup-only throughput
    # ax = axs[0]
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(lookuponly_throughput[idx].get(task, 0))
    #     ax.bar([x + i*bar_width for x in index], task_data, bar_width, label=task, color=colors[i])
    #
    # ax.set_title('Lookup-only Throughput')
    # ax.set_ylabel('Throughput (Mops/s)')
    # ax.set_xticks([x + bar_width*1.5 for x in index])
    # ax.set_xticklabels(indexs)
    # ax.legend()
    #
    # # 2. Plot insert-lookup throughput (separated)
    # ax = axs[1]
    # # First plot lookups
    # offset = 0
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(insertlookup_throughput[idx]['lookup'].get(task, 0))
    #     ax.bar([x + offset for x in index], task_data, bar_width/2,
    #            label=f'{task} (lookup)' if offset == 0 else "_nolegend_",
    #            color=colors[i])
    #     offset += bar_width/2
    #
    # # Then plot inserts
    # offset = bar_width*2
    # for i, task in enumerate(tasks):
    #     task_data = []
    #     for idx in indexs:
    #         task_data.append(insertlookup_throughput[idx]['insert'].get(task, 0))
    #     ax.bar([x + offset for x in index], task_data, bar_width/2,
    #            label=f'{task} (insert)', color=colors[i], hatch='///')
    #     offset += bar_width/2
    #
    # ax.set_title('Insert-Lookup Throughput (50% insert ratio)')
    # ax.set_ylabel('Throughput (Mops/s)')
    # ax.set_xticks([x + bar_width*1.5 for x in index])
    # ax.set_xticklabels(indexs)
    # ax.legend()
    
    # 3. Plot mixed workload with 10% inserts
    ax = axs[0]
    for i, task in enumerate(tasks):
        task_data = []
        for idx in indexs:
            task_data.append(insertlookup_mix1_throughput[idx].get(task, 0))
        ax.bar([x + i*bar_width for x in index], task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Mixed Workload (10% insert ratio)')
    ax.set_ylabel('Throughput (Mops/s)')
    ax.set_xticks([x + bar_width*1.5 for x in index])
    ax.set_xticklabels(indexs)
    ax.legend()
    
    # 4. Plot mixed workload with 90% inserts
    ax = axs[1]
    for i, task in enumerate(tasks):
        task_data = []
        for idx in indexs:
            task_data.append(insertlookup_mix2_throughput[idx].get(task, 0))
        ax.bar([x + i*bar_width for x in index], task_data, bar_width, label=task, color=colors[i])
        
    ax.set_title('Mixed Workload (90% insert ratio)')
    ax.set_ylabel('Throughput (Mops/s)')
    ax.set_xticks([x + bar_width*1.5 for x in index])
    ax.set_xticklabels(indexs)
    ax.legend()
    
    # Add overall title and adjust layout
    fig.suptitle('Benchmark Results Across Different Workloads', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()
    
    # Save data to CSV files for further analysis
    import os
    os.makedirs('analysis_results', exist_ok=True)
    
    # pd.DataFrame(lookuponly_throughput).to_csv('analysis_results/lookuponly_throughput.csv')
    
    # lookup_df = pd.DataFrame({idx: data['lookup'] for idx, data in insertlookup_throughput.items()})
    # insert_df = pd.DataFrame({idx: data['insert'] for idx, data in insertlookup_throughput.items()})
    # lookup_df.to_csv('analysis_results/insertlookup_lookup_throughput.csv')
    # insert_df.to_csv('analysis_results/insertlookup_insert_throughput.csv')
    #
    pd.DataFrame(insertlookup_mix1_throughput).to_csv('analysis_results/insertlookup_mix1_throughput.csv')
    pd.DataFrame(insertlookup_mix2_throughput).to_csv('analysis_results/insertlookup_mix2_throughput.csv')

if __name__ == "__main__":
    result_analysis()
        


        
        
        
        
    