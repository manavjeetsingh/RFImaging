from gnuradio import gr, blocks
import osmosdr

class CW_TX(gr.top_block):
 def __init__(self, f=915e6, g=10):
  gr.top_block.__init__(self, "TX")
  # src: const DC; snk: bladeRF
  s = self.sk = osmosdr.sink("bladerf=0,buffers=128,buflen=8192")
  s.set_sample_rate(1e6)
  s.set_center_freq(f, 0)
  s.set_gain(g, 0)
  self.connect(blocks.vector_source_c([1+0j],True), s)

 def set_f(self, f): # upd freq
  self.sk.set_center_freq(f*1e6, 0)

def set_g(self, g):
    self.sk.set_gain(g, 0)

if __name__ == "__main__":
 tb = CW_TX()
 tb.start()
 try:
  while 1:
   i = input("Freq (MHz) [q:exit]: ")
   if i == 'q': break
   tb.set_f(float(i)) # apply
 except: pass
 tb.stop(); tb.wait()