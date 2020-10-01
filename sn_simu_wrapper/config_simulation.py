from sn_tools.sn_io import recursive_merge

class ConfigSimulation:
    """
    class to load a set of parameters (txt file)
    and make a dict out of them

    Parameters
    --------------
    type_sn: str
       SN type
    model_sn: str
      model for simulation
    config_file: str
      configuration file

    """
    

    def __init__(self,type_sn,model_sn,config_file):

        # first: load all possible parameters
        cdict = self.configDict(config_file)

        # second: keep keys only valids for (type_sn, model_sn)

        self.conf_dict = self.select(cdict,type_sn,model_sn)
        
    def configDict(self,config_file):
        """
        Method to load a txt configuration file
        and makes a dict out of it

        Parameters
        --------------
        config_file: str
          config file (txt) to transform to a dict

        Returns
        ----------
        the dict

        """
        
        ffile = open(config_file, 'r') 
        line = ffile.read().splitlines()
        ffile.close()
        
        params = {}
        for i,ll in enumerate(line):
            if ll!='' and ll[0]!='#':
                lspl = ll.split(' ')
                n = len(lspl)
                mystr = ''
                myclose = ''
                for keya in [lspl[i] for i in range(n-2)]:
                    mystr += '{\''+keya+ '\':'
                    myclose +=' }'

                if lspl[n-1]!= 'str':
                    dd = '{} {} {}'.format(mystr,eval('{}({})'.format(lspl[n-1],lspl[n-2])),myclose)
                else:
                   dd = '{} \'{}\' {}'.format(mystr,lspl[n-2],myclose)

                thedict = eval(dd)
                params = recursive_merge(params, thedict)

        return params

    def select(self, thedict, type_sn, model_sn):
        """
        Method to select thedict keys according to type_sn and model_sn

        Parameters
        ---------------
        thedict: dict
          original dict
        type_sn: str
          SN type
        model_sn: str
          SN model for simulation
        """

        res = dict(thedict)
        if 'Ia' not in type_sn:
            del res['SN']['x1']
            del res['SN']['color']
            del res['SN']['x1_color']

        return res
        
