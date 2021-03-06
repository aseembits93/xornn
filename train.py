from common_utils import *
import ipdb as pdb

def train(train_loader, model, criterion, optimizer, epoch, device, args=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        #top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (X,Y) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                X = X.to(device)
                Y = Y.to(device)
                # compute output
                output = model(X)
                loss = criterion(output, Y)
                # measure accuracy and record loss
                acc1 = accuracy(output, Y, topk=(1,2))
                losses.update(loss, X.size(0))
                top1.update(acc1[0], X.size(0))
                #top5.update(acc5[0], input.size(0))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
        print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time))
        print('Loss : '+str(losses.avg.item()))    
        print('Accuracy : '+str(float(top1.avg)))


def train_LSTM(train_loader, model, criterion, optimizer, epoch, device, args=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        #top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (X,Y) in enumerate(train_loader):
                #randomize hidden state before forward pass
                #hd = [torch.zeros(4,,10),torch.zeros(4,32,10)]
                # measure data loading time
                data_time.update(time.time() - end)
                X = X.to(device)
                Y = Y.to(device)
                #hd = [l.to(device) for l in hd]
                #pdb.set_trace()
                # compute output
                output = model(X)#,hd)
                #pdb.set_trace()
                loss = criterion(output, Y)
                # measure accuracy and record loss
                acc1 = accuracy(output, Y, topk=(1,2))
                losses.update(loss, X.size(0))
                top1.update(acc1[0], X.size(0))
                #top5.update(acc5[0], input.size(0))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
        print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time))
        print('Loss : '+str(losses.avg.item()))    
        print('Accuracy : '+str(float(top1.avg)))
# def validate(val_loader, model, criterion, args):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         for i, (input, target) in enumerate(val_loader):
#             if args.gpu is not None:
#                 input = input.cuda(args.gpu, non_blocking=True)
#             target = target.cuda(args.gpu, non_blocking=True)

#             # compute output
#             output = model(input)
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), input.size(0))
#             top1.update(acc1[0], input.size(0))
#             top5.update(acc5[0], input.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                     'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                     'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     i, len(val_loader), batch_time=batch_time, loss=losses,
#                     top1=top1, top5=top5))

#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#             .format(top1=top1, top5=top5))

#     return top1.avg
