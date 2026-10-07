Revision SFSDP122 from SFSDP111
                                                     January 2010

This new version improves the perfomance of SFSDP. 



Revision SFSDP111 from SFSDP100
                                                        July 2009

As a default, this new version of SFSDP calls sedumiwrap, which is a 
MATLAB version of SDPA, to solve an SDP relaxation problem. If users would 
like to use SeDuMi instead of SDPA, they need to modify the MATLAB program 
SFSDP.m by replacing the line 

SDPsolverDefault = 'sdpa'; 

by 

SDPsolverDefault = 'sedumi'; 

Or users can set a new parameter pars.SDPsolver = 'sedumi' at each 
execution of SFSDPplus or SFSDP. 
 