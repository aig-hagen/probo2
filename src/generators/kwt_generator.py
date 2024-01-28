import random
import networkx as nx
from itertools import repeat
import time
class KwtDungTheoryGenerator():
    """ generated source for class KwtDungTheoryGenerator """
    # 
    # 	 * 
    # 	 * @param num_arguments nr of args
    # 	 * @param num_skept_arguments nr of skeptical args
    # 	 * @param size_ideal_extension size of ideal extension
    # 	 * @param num_cred_arguments nr of credulous args
    # 	 * @param num_pref_exts nums of preferred extensions
    # 	 * @param p_ideal_attacked idel attacked
    # 	 * @param p_ideal_attack_back ideal aattcked back
    # 	 * @param p_other_skept_args_attacked skeptical attacked back
    # 	 * @param p_other_skept_args_attack_back skeptical attacked args back
    # 	 * @param p_cred_args_attacked credulous args attacked
    # 	 * @param p_cred_args_attack_back creduous args attack back
    # 	 * @param p_other_attacks other attack
    # 	 
    def __init__(self, num_arguments, num_skept_arguments, size_ideal_extension, num_cred_arguments, num_pref_exts, p_ideal_attacked, p_ideal_attack_back, p_other_skept_args_attacked, p_other_skept_args_attack_back, p_cred_args_attacked, p_cred_args_attack_back, p_other_attacks):
        """ generated source for method __init__ """
        super(KwtDungTheoryGenerator, self).__init__()
        self.num_arguments = num_arguments
        self.num_skept_arguments = num_skept_arguments
        self.size_ideal_extension = size_ideal_extension
        self.num_cred_arguments = num_cred_arguments
        self.num_pref_exts = num_pref_exts
        self.p_ideal_attacked = p_ideal_attacked
        self.p_ideal_attack_back = p_ideal_attack_back
        self.p_other_skept_args_attacked = p_other_skept_args_attacked
        self.p_other_skept_args_attack_back = p_other_skept_args_attack_back
        self.p_cred_args_attacked = p_cred_args_attacked
        self.p_cred_args_attack_back = p_cred_args_attack_back
        self.p_other_attacks = p_other_attacks
        
    def generate_instance(self):
        """ generated source for method next """
        st = time.time()
        af = nx.DiGraph()
        args = [ f'a{i}' for i in range(0,self.num_arguments-1)]
        af.add_nodes_from(args)
        skept_args = args[:self.num_skept_arguments]
        cred_args = args[self.num_skept_arguments:self.num_skept_arguments +self.num_cred_arguments]
        pref_exts = [ skept_args.copy() for i in range(0,self.num_pref_exts)]
        ideal_ext = skept_args[:self.size_ideal_extension]
        other_skept_args = skept_args[self.size_ideal_extension :]
        unaccepted_arguments = args[self.num_skept_arguments + self.num_cred_arguments:]
        et = time.time() 
        elapsed_time = et - st
        #print(f'Init + Slicing:{elapsed_time} seconds')
        # print(nx.info(af))
        # print(f'{args=} Size: {len(args)}')

        # print(f'{skept_args=} Size: {len(skept_args)}')
        # print(f'{cred_args=} Size: {len(cred_args)}')
        # print(f'{pref_exts=} Size: {len(pref_exts)}')
        # print(f'{ideal_ext=} Size: {len(ideal_ext)}')
        # print(f'{other_skept_args=} Size: {len(other_skept_args)}')
        # print(f'{unaccepted_arguments=} Size: {len(unaccepted_arguments)}')

        st = time.time()
        for cred_arg in cred_args:
           
            num_preffered_extension_to_include_argument = random.randint(1,self.num_pref_exts -1)
           
            index_pref_extensions = random.sample(range(0,self.num_pref_exts),num_preffered_extension_to_include_argument)
           
            for index in index_pref_extensions:
                
                cur_extension = pref_exts[index]
               
                cur_extension.append(cred_arg)
        et = time.time() 
        elapsed_time = et - st
       # print(f'Add Cred. Args:{elapsed_time} seconds')      
        
        #print(f'{pref_exts=}')
    

        # all arguments in the ideal extension should be attacked
		# by some unaccepted argument (in order to have an empty
		# grounded extension) and defended by the ideal extension
        st = time.time()
        for arg_ideal in ideal_ext:
            for arg_unaccepted in unaccepted_arguments:
                if random.random() < self.p_ideal_attacked: 
                    af.add_edge(arg_unaccepted,arg_ideal)
                    for arg_ideal_back in ideal_ext:
                        if random.random() < self.p_ideal_attack_back:
                            af.add_edge(arg_ideal_back,arg_unaccepted)
        et = time.time() 
        elapsed_time = et - st
        #print(f'IDEAL:{elapsed_time} seconds')   
				
        #print(nx.info(af))

        st= time.time()
        for other_skpet_arg  in other_skept_args: 
            for unaccepted_arg in unaccepted_arguments:
            	if random.random() < self.p_other_skept_args_attacked:
                    af.add_edge(unaccepted_arg,other_skpet_arg)
                    for extension in  pref_exts:					
                        for pref_arg in extension:
                            if pref_arg not in ideal_ext and pref_arg not in other_skept_args:
                                if random.random() < self.p_other_skept_args_attack_back:
                                    af.add_edge(pref_arg,unaccepted_arg)     
        et = time.time() 
        elapsed_time = et - st
        #print(f'Other Skep:{elapsed_time} seconds')    

        #print(nx.info(af))
        # for every pref. extension and every not skeptically accepted
		# argument in there, it should be attacked from outside and defended
        for pref_ext in pref_exts:
            for pref_arg in pref_ext:
                if pref_arg not in ideal_ext and pref_arg not in other_skept_args:
                    for arg in args:
                        if arg not in ideal_ext and arg not in other_skept_args and arg not in pref_ext:
                            if random.random() < self.p_cred_args_attacked:
                                af.add_edge(arg,pref_arg)
                            for o_pref_arg in pref_ext:
                                if o_pref_arg not in ideal_ext and o_pref_arg not in other_skept_args:
                                    if random.random() < self.p_cred_args_attack_back:
                                        af.add_edge(o_pref_arg, arg)		

        #print(nx.info(af))
        # add some other attacks between unaccepted arguments
        for a in unaccepted_arguments:
            for b in unaccepted_arguments:
                if random.random() < self.p_other_attacks:
                    af.add_edge(a,b)
        
        #print(nx.info(af))

        # add something to increase the likelihood of having 
		# no stable extensions
        af.add_node('c')
        af.add_edge('c','c')
        for a in unaccepted_arguments:
            if bool(random.getrandbits(1)):
                if bool(random.getrandbits(1)):
                    af.add_edge(a,'c')
                else:
                    af.add_edge('c',a)
        return af
    


if __name__ == '__main__':
    generator = KwtDungTheoryGenerator(1000,50,3,3,5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
    for i in range(0,10):
        generator.generate_instances()